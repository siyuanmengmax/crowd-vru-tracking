"""Self-training baseline (protocol in the paper, Self-Training Baseline): pseudo-labelled training samples from unlabeled LUMPI M6 frames outside the
held-out clips (100-frame buffer), every 9th frame, one sample per LiDAR. The teacher is the zero-shot VoxelNeXt with the
exact per-LiDAR input of lumpi_det.py; a sample stores that model input (prepare_cloud output) and the teacher's own boxes
on it with score >= 0.45 that contain >= 1 foreground return (M6 background model, all LiDARs), in the model frame
(z minus z_sensor_offset), velocity 0. No label is read. Run with the M6 environment (LSRC, LSENS, LCACHE=_m6).
Writes st_pkg/ft_data/m6/<frame>_<lidar>.npy and st_pkg/ft_data/infos_train.pkl (ft_train.py format)."""
import os, pickle, time
import numpy as np, pandas as pd
from det_v2x import PretrainedDetector, CFG, CKPT, HERE
from lumpi_prep import ply, xyz, sensors, background, occ, THETA
from lumpi_det import sensor_ground
OUT = f"{HERE}/st_pkg/ft_data"; STEP, BUF, SCORE = 9, 100, 0.45

def frames():
    starts = pd.read_csv(f"{HERE}/m6/clips_m6.csv").start_frame.tolist()
    n = len(os.listdir(f"{os.environ['LSRC']}/lidar"))
    bad = np.zeros(n, bool)
    for s in starts: bad[max(0, s - BUF):s + 300 + BUF] = True
    return [f for f in range(0, n, STEP) if not bad[f]]

def in_box(P, b):   # b = x, y, z, l, w, h, yaw (world frame)
    c, s = np.cos(b[6]), np.sin(b[6]); dx, dy = P[:, 0] - b[0], P[:, 1] - b[1]
    return (np.abs(c * dx + s * dy) <= b[3] / 2) & (np.abs(-s * dx + c * dy) <= b[4] / 2) & (np.abs(P[:, 2] - b[2]) <= b[5] / 2)

if __name__ == "__main__":
    S = sensors(); G = sensor_ground(); BG = background(); os.makedirs(f"{OUT}/m6", exist_ok=True)
    det = PretrainedDetector(CFG, CKPT, score_thresh=0.1, point_cloud_range=[-54, -54, -5, 54, 54, 3]); zoff = det.z_sensor_offset
    F = frames(); print("frames", len(F), "first", F[:3], flush=True); infos = []; kept = {}; t0 = time.time()
    for n, src in enumerate(F):
        p = ply(src); X = xyz(p); FG = X[occ(BG, X) < THETA]; clouds, offs, sids = [], [], []
        for s, o in S.items():
            m = p["id"] == s
            if not m.any(): continue
            off = np.array([o[0], o[1], G[s]]); q = X[m] - off
            k = (np.abs(q[:, 0]) < 53.5) & (np.abs(q[:, 1]) < 53.5) & (q[:, 2] > -2.9) & (q[:, 2] < 4.5); m = np.flatnonzero(m)[k]; q = q[k]
            arr = np.column_stack([q, np.linalg.norm(X[m] - o, axis=1), p["intensity"][m].astype(np.float64)])
            pts, _ = det.prepare_cloud(arr, ["x", "y", "z", "range", "reflectivity"]); clouds.append(pts); offs.append(off); sids.append(s)
        for pts, off, s, r in zip(clouds, offs, sids, det.infer(clouds)):
            boxes, names = [], []
            for b, sc, l in zip(r["boxes"], r["scores"], r["labels"]):
                if sc < SCORE: continue
                w = b.copy(); w[:3] += off
                if not in_box(FG, w).any(): continue
                boxes.append([b[0], b[1], b[2] - zoff, b[3], b[4], b[5], b[6], 0.0, 0.0]); names.append(l); kept[l] = kept.get(l, 0) + 1
            fn = f"{OUT}/m6/{src:06d}_{s}.npy"; np.save(fn, pts.astype(np.float32))
            infos.append(dict(frame_id=f"m6/{src:06d}_{s}", lidar_path=fn, gt_boxes=np.array(boxes, np.float32).reshape(-1, 9), gt_names=np.array(names)))
        if n % 100 == 0: print(n, len(F), f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(infos, open(f"{OUT}/infos_train.pkl", "wb"))
    print("samples", len(infos), "pseudo-boxes", kept, "ST_PREP_DONE", flush=True)
