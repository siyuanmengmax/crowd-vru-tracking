"""Supervised upper bound: the fine-tuned detectors (cloud run of ft_train.py) on the 15 val/test scenarios.
Same input frame as training (ft_prep.py): levelled with the LiDAR pose, centred on the LiDAR in x-y, z relative to the
label-free local ground under the LiDAR (+0.19) minus the nuScenes sensor height; each LiDAR inferred on its own, then
late-fused in the world frame exactly as det_v2x.py. Writes work_vru/<key>/det_ft_<ARCH>.csv (boxes scoring >= 0.1).
Usage: ARCH=voxelnext|centerpoint [EP=8] python det_ft.py"""
import glob, os, time
import numpy as np, pandas as pd, yaml
from det_v2x import pose_matrix, SRC, PretrainedDetector, late_fuse, WORK, HERE
from ground_map import ground_z
ARCH = os.environ.get("ARCH", "voxelnext"); EP = int(os.environ.get("EP", 8))
CFG = {"voxelnext": "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml", "centerpoint": "cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml"}[ARCH]
CKPT = os.environ.get("FT_CKPT", f"{HERE}/../checkpoints/ft_{ARCH}_ep{EP}.pth"); OUT = os.environ.get("DET_NAME", f"det_ft_{ARCH}")

def run(det, path):
    key = "__".join(path.rstrip("/").split("/")[-2:]); rows = []
    for i, yf in enumerate(sorted(glob.glob(f"{path}/-1/*.yaml"))):
        clouds, mats = [], []
        for ag in ("-1", "-2"):
            f = yf.replace("/-1/", f"/{ag}/")
            if not os.path.exists(f): continue
            y = yaml.load(open(f), Loader=yaml.CSafeLoader)
            p = np.fromfile(f.replace(".yaml", ".bin"), np.float32).reshape(-1, 4); p = p[np.abs(p[:, :3]).sum(1) > 0.5]
            M = pose_matrix(y["lidar_pose"]); w = p[:, :3] @ M[:3, :3].T + M[:3, 3]
            off = np.array([M[0, 3], M[1, 3], float(ground_z(M[0, 3], M[1, 3])) + 0.19])   # prepare_cloud subtracts the 1.84
            arr = np.column_stack([w - off, np.linalg.norm(p[:, :3], axis=1), np.clip(p[:, 3], 0, 255)])
            pts, _ = det.prepare_cloud(arr, ["x", "y", "z", "range", "reflectivity"])
            T = np.eye(4); T[:3, 3] = off; clouds.append(pts); mats.append(T)
        fused = late_fuse(det.infer(clouds), mats)
        for b, s, l, n in zip(fused["boxes"], fused["scores"], fused["labels"], fused["num_sensors"]):
            rows.append(dict(frame_id=i, x=b[0], y=b[1], z=b[2], l=b[3], w=b[4], h=b[5], yaw=b[6], score=s, cls=l, num_sensors=n))
    pd.DataFrame(rows).to_csv(f"{WORK}/{key}/{OUT}.csv", index=False)
    return key, len(rows)

if __name__ == "__main__":
    det = PretrainedDetector(CFG, CKPT, score_thresh=0.1)   # training used the config's own point-cloud range
    paths = [p for s in ("val", "test") for p in sorted(glob.glob(f"{SRC}/{s}/*/")) if os.path.isdir(p + "-1")]   # never the train splits
    for p in paths:
        key = "__".join(p.rstrip("/").split("/")[-2:])
        if os.path.exists(f"{WORK}/{key}/{OUT}.csv"): continue
        t = time.time(); print(run(det, p), f"{time.time() - t:.0f}s", flush=True)
