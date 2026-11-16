"""Zero-shot nuScenes VoxelNeXt on the LUMPI clips, exactly as on V2X-Real (det_v2x.py) and label-free: each of the five
LiDARs is inferred on its own (its returns centred on the sensor in x-y, z relative to the ground under the sensor),
then the boxes are late-fused in the common frame (st_pkg/fusion.py late_fuse).
Ground under a sensor = height mode of the returns 2-10 m around it (label-free; LUMPI is flat, so no ground map; the
V2X-Real voxel rule + 0.19 m sits ~0.15 m above the LUMPI ground returns).  DET=ft_voxelnext uses the V2X-Real fine-tuned
checkpoint instead (cross-site transfer of the supervised upper bound).
Writes work_lumpi/<clip>/det_voxelnext.csv (or det_ft_voxelnext.csv), all boxes scoring >= 0.1."""
import glob, os, time
import numpy as np, pandas as pd
from det_v2x import PretrainedDetector, late_fuse, CFG, CKPT, HERE
from lumpi_prep import ply, xyz, sensors, WORK
DET = os.environ.get("DET", "voxelnext")

def sensor_ground(frames=(100, 2000, 4000, 6000, 8000)):
    """Height of the ground returns around each sensor: mode (2 cm bins) of all returns 2-10 m away in a few frames."""
    S = sensors(); X = np.vstack([xyz(ply(f)) for f in frames]); out = {}
    for s, o in S.items():
        r = np.hypot(X[:, 0] - o[0], X[:, 1] - o[1]); z = X[(r > 2) & (r < 10), 2]
        h, e = np.histogram(z, bins=np.arange(o[2] - 4, o[2], 0.02)); out[s] = float(e[h.argmax()] + 0.01)
    return out

def run(det, d, G, S):
    rows = []; frames = np.load(f"{d}/frames.npy")
    for i, src in enumerate(frames):
        p = ply(int(src)); X = xyz(p); clouds, mats = [], []
        for s, o in S.items():
            m = p["id"] == s
            if not m.any(): continue
            off = np.array([o[0], o[1], G[s]]); q = X[m] - off
            k = (np.abs(q[:, 0]) < 53.5) & (np.abs(q[:, 1]) < 53.5) & (q[:, 2] > -2.9) & (q[:, 2] < 4.5); m = np.flatnonzero(m)[k]; q = q[k]   # strictly inside the grid
            arr = np.column_stack([q, np.linalg.norm(X[m] - o, axis=1), p["intensity"][m].astype(np.float64)])
            pts, _ = det.prepare_cloud(arr, ["x", "y", "z", "range", "reflectivity"])
            T = np.eye(4); T[:3, 3] = off; clouds.append(pts); mats.append(T)
        fused = late_fuse(det.infer(clouds), mats)
        for b, sc, l, n in zip(fused["boxes"], fused["scores"], fused["labels"], fused["num_sensors"]):
            rows.append(dict(frame_id=i, x=b[0], y=b[1], z=b[2], l=b[3], w=b[4], h=b[5], yaw=b[6], score=sc, cls=l, num_sensors=n))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    S = sensors(); G = sensor_ground(); print("ground under sensors", {k: round(v, 2) for k, v in G.items()})
    if DET == "voxelnext": det, name = PretrainedDetector(CFG, CKPT, score_thresh=0.1, point_cloud_range=[-54, -54, -5, 54, 54, 3]), "det_voxelnext"
    elif DET == "st_voxelnext": det, name = PretrainedDetector(CFG, os.environ.get("ST_CKPT", f"{HERE}/../checkpoints/st_voxelnext_ep8.pth"), score_thresh=0.1), "det_st_voxelnext"   # self-training baseline
    else: det, name = PretrainedDetector(CFG, os.environ.get("FT_CKPT", f"{HERE}/../checkpoints/ft_voxelnext_ep8.pth"), score_thresh=0.1), "det_ft_voxelnext"
    for d in sorted(glob.glob(f"{WORK}/lumpi__*")):
        if os.path.exists(f"{d}/{name}.csv"): continue
        t = time.time(); df = run(det, d, G, S); df.to_csv(f"{d}/{name}.csv", index=False)
        print(os.path.basename(d), len(df), f"{time.time() - t:.0f}s", flush=True)
