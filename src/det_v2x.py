"""Pre-check B: a learned detector (nuScenes VoxelNeXt, zero-shot) on the V2X-Real infrastructure LiDARs.
Each LiDAR is inferred on its own (points levelled with the LiDAR pose, centred on the LiDAR, ground at z = 0
from the median bottom of the annotated boxes, or with FLAT=1 from the label-free local ground map), then the two are late-fused in the world frame
(st_pkg/fusion.py late_fuse). Writes work_vru/<key>/det_voxelnext.csv with all boxes scoring >= 0.1."""
import glob, os, pickle, sys, time
import numpy as np, pandas as pd, yaml
from v2x_prep import pose_matrix, SRC
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_pkg"))
from detector import PretrainedDetector
from fusion import late_fuse
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru")
CFG = "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
CKPT = os.environ.get("VOXELNEXT_CKPT", os.path.join(HERE, "..", "checkpoints", "cbgs_voxel0075_voxelnext_nds6665.pth"))
OUT = os.environ.get("DET_NAME", "det_voxelnext"); FLAT = int(os.environ.get("FLAT", 0))   # FLAT: heights relative to the local ground map
from ground_map import ground_z as local_ground

def ground_z(gt):
    z = [b["z"] - b["h"] / 2 for v in gt.values() for b in v if b["category_id"] in (0, 1)]
    return float(np.median(z))

def run(det, path):
    key = "__".join(path.rstrip("/").split("/")[-2:])
    gt = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb")); g0 = ground_z(gt); rows = []
    for i, yf in enumerate(sorted(glob.glob(f"{path}/-1/*.yaml"))):
        clouds, mats = [], []
        for ag in ("-1", "-2"):
            f = yf.replace("/-1/", f"/{ag}/")
            if not os.path.exists(f): continue
            y = yaml.load(open(f), Loader=yaml.CSafeLoader)
            p = np.fromfile(f.replace(".yaml", ".bin"), np.float32).reshape(-1, 4); p = p[np.abs(p[:, :3]).sum(1) > 0]
            M = pose_matrix(y["lidar_pose"]); w = p[:, :3] @ M[:3, :3].T + M[:3, 3]
            off = np.array([M[0, 3], M[1, 3], g0])
            q = w - off
            if FLAT: q[:, 2] = w[:, 2] - (local_ground(w[:, 0], w[:, 1]) + 0.19)   # label-free, the site slopes ~5 m
            arr = np.column_stack([q, np.linalg.norm(p[:, :3], axis=1), np.clip(p[:, 3], 0, 255)])
            pts, _ = det.prepare_cloud(arr, ["x", "y", "z", "range", "reflectivity"])
            T = np.eye(4); T[:3, 3] = off; clouds.append(pts); mats.append(T)
        fused = late_fuse(det.infer(clouds), mats)
        for b, s, l, n in zip(fused["boxes"], fused["scores"], fused["labels"], fused["num_sensors"]):
            z = b[2] - g0 + float(local_ground(b[0], b[1])) + 0.19 if FLAT else b[2]
            rows.append(dict(frame_id=i, x=b[0], y=b[1], z=z, l=b[3], w=b[4], h=b[5], yaw=b[6], score=s, cls=l, num_sensors=n))
    pd.DataFrame(rows).to_csv(f"{WORK}/{key}/{OUT}.csv", index=False)
    return key, len(rows)

if __name__ == "__main__":
    det = PretrainedDetector(CFG, CKPT, score_thresh=0.1, point_cloud_range=[-54, -54, -5, 54, 54, 3])
    paths = [p for p in sorted(glob.glob(f"{SRC}/*/*/")) if os.path.isdir(p + "-1")]
    if os.environ.get("LIMIT"): paths = paths[:int(os.environ["LIMIT"])]
    for p in paths:
        key = "__".join(p.rstrip("/").split("/")[-2:])
        if os.path.exists(f"{WORK}/{key}/{OUT}.csv"): continue
        t = time.time(); print(run(det, p), f"{time.time() - t:.0f}s", flush=True)
