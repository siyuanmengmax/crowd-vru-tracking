"""Detector-only baseline (tags det0.3, det0.45): VoxelNeXt boxes go straight into the trackers' box CSV.
Class: pedestrian/bicycle/motorcycle -> 0, car -> 1, truck/bus/trailer/construction_vehicle -> 2.
members.pkl (evaluation only) lists the GT objects owning the real-foreground returns inside each box."""
import glob, os, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
from split_prep import owners
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru")
DET = os.environ.get("DET", "det_voxelnext"); PFX = os.environ.get("PFX", "det")   # e.g. DET=det_ft_voxelnext PFX=ftvxn
CLS = {"pedestrian": 0, "bicycle": 0, "motorcycle": 0, "car": 1, "truck": 2, "bus": 2, "trailer": 2, "construction_vehicle": 2}

def in_box(p, b):
    c, s = np.cos(b.yaw), np.sin(b.yaw); dx, dy = p[:, 0] - b.x, p[:, 1] - b.y
    return (np.abs(c * dx + s * dy) < b.l / 2 + 0.1) & (np.abs(-s * dx + c * dy) < b.w / 2 + 0.1) & (np.abs(p[:, 2] - b.z) < b.h / 2 + 0.1)

def one(key):
    gt = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb")); det = pd.read_csv(f"{WORK}/{key}/{DET}.csv")
    det = det[det.cls.isin(CLS)]
    for t in (0.3, 0.45):
        tag = f"{PFX}{t}"; os.makedirs(f"{WORK}/{key}/{tag}/detection", exist_ok=True); rows, mem = [], []
        for fid, g in det[det.score >= t].groupby("frame_id"):
            f = f"{WORK}/{key}/real0.5/detection/{fid:06d}.bin"
            pts = np.fromfile(f, np.float32).reshape(-1, 10).astype(np.float64); own = owners(pts, gt[fid]) if len(pts) else np.zeros(0, int)
            for cid, b in enumerate(g.itertuples()):
                ok = own[in_box(pts, b)] if len(pts) else own; ok = ok[ok >= 0]
                rows.append(dict(frame_id=fid, frame_name=f"{fid:06d}.bin", cluster_id=cid, label=CLS[b.cls], confidence=b.score,
                                 x=b.x, y=b.y, z=b.z, w=b.l, l=b.w, h=b.h, yaw=b.yaw, avg_eps=0.0, avg_reliability=1.0, num_points=len(ok)))
                mem.append(dict(frame_id=fid, cluster_id=cid, owners={gt[fid][j]["track_id"]: int((ok == j).sum()) for j in np.unique(ok)}))
        pd.DataFrame(rows).to_csv(f"{WORK}/{key}/{tag}/detection_bbox_results.csv", index=False)
        pickle.dump(mem, open(f"{WORK}/{key}/{tag}/members.pkl", "wb"))
    return key

if __name__ == "__main__":
    keys = sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/*__*") if os.path.exists(f"{p}/{DET}.csv"))
    with Pool(24) as p: print(len(p.map(one, keys)))
