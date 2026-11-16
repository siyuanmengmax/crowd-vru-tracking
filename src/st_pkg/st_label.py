"""Self-training baseline (protocol in the paper, Self-Training Baseline), GPU part, run in the cloud.
1. Teacher check: the zero-shot VoxelNeXt (same settings as lumpi_det.py) on 30 frames of an M6 evaluation clip, late-fused
   as there, compared with the stored workstation output (st_check/det_voxelnext_ref.csv); stops if >= 1% of the stored
   boxes have no box within 1 cm (the rule used for every GPU check in this study). These frames are not used for training.
2. Pseudo-labels on st_in/<frame>.npz (unlabeled M6 frames outside the held-out clips): per LiDAR, the teacher's boxes with
   score >= 0.45 that contain >= 1 foreground return, in the model frame, velocity 0 -> ft_data/m6/*.npy + infos_train.pkl."""
import glob, os, pickle, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
P = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, P)
from detector import PretrainedDetector
from fusion import late_fuse
CFG = "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"; SCORE = 0.45
COLS = ["x", "y", "z", "range", "reflectivity"]

def in_box(Q, b):
    c, s = np.cos(b[6]), np.sin(b[6]); dx, dy = Q[:, 0] - b[0], Q[:, 1] - b[1]
    return (np.abs(c * dx + s * dy) <= b[3] / 2) & (np.abs(-s * dx + c * dy) <= b[4] / 2) & (np.abs(Q[:, 2] - b[2]) <= b[5] / 2)

def inputs(det, z):
    clouds, offs = [], []
    for k in sorted(x for x in z.files if x.startswith("arr_")):
        s = k[4:]; pts, _ = det.prepare_cloud(z[k].astype(np.float64), COLS); clouds.append(pts); offs.append((int(s), z[f"off_{s}"]))
    return clouds, offs

if __name__ == "__main__":
    det = PretrainedDetector(CFG, f"{P}/ckpt/cbgs_voxel0075_voxelnext_nds6665.pth", openpcdet_root=f"{P}/OpenPCDet",
                             score_thresh=0.1, point_cloud_range=[-54, -54, -5, 54, 54, 3]); zoff = det.z_sensor_offset
    ref = pd.read_csv(f"{P}/st_check/det_voxelnext_ref.csv"); un = 0
    for src, R in ref.groupby("src"):
        clouds, offs = inputs(det, np.load(f"{P}/st_check/{src:06d}.npz")); mats = []
        for _, off in offs: T = np.eye(4); T[:3, 3] = off; mats.append(T)
        B = late_fuse(det.infer(clouds), mats)["boxes"]
        un += len(R) if not len(B) else int((cKDTree(np.asarray(B)[:, :3]).query(R[["x", "y", "z"]].to_numpy())[0] >= 0.01).sum())
    share = un / len(ref); print(f"TEACHER CHECK {100 * share:.2f}% of {len(ref)} stored boxes unmatched", "OK" if share < 0.01 else "MISMATCH", flush=True)
    if share >= 0.01: sys.exit(1)
    out = f"{P}/ft_data/m6"; os.makedirs(out, exist_ok=True); infos = []; kept = {}
    for n, f in enumerate(sorted(glob.glob(f"{P}/st_in/*.npz"))):
        z = np.load(f); FG = z["fg"].astype(np.float64); clouds, offs = inputs(det, z); src = os.path.basename(f)[:6]
        for pts, (s, off), r in zip(clouds, offs, det.infer(clouds)):
            boxes, names = [], []
            for b, sc, l in zip(r["boxes"], r["scores"], r["labels"]):
                if sc < SCORE: continue
                w = np.array(b, np.float64); w[:3] += off
                if not in_box(FG, w).any(): continue
                boxes.append([b[0], b[1], b[2] - zoff, b[3], b[4], b[5], b[6], 0.0, 0.0]); names.append(l); kept[l] = kept.get(l, 0) + 1
            fn = f"{out}/{src}_{s}.npy"; np.save(fn, pts.astype(np.float32))
            infos.append(dict(frame_id=f"m6/{src}_{s}", lidar_path=fn, gt_boxes=np.array(boxes, np.float32).reshape(-1, 9), gt_names=np.array(names)))
        if n % 100 == 0: print(n, flush=True)
    pickle.dump(infos, open(f"{P}/ft_data/infos_train.pkl", "wb"))
    print("samples", len(infos), "pseudo-boxes", kept, "ST_LABEL_DONE", flush=True)
