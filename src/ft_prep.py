"""Training data for the site-specific upper bound: VoxelNeXt (nuScenes weights) fine-tuned on the V2X-Real
infrastructure train split (train1-4, 43 scenarios). One sample per LiDAR and every STEP-th frame, in the same
input frame the detector sees at inference: levelled with the LiDAR pose, centred on the LiDAR in x-y, z relative to
the label-free local ground under the LiDAR (ground_map.py) minus the nuScenes sensor height 1.84 m.
Boxes: annotated objects with >= MIN_PTS returns of THIS LiDAR, mapped to the nuScenes classes the head already has
(ScooterRider -> pedestrian); velocity from the annotated track (10 Hz). Writes ft_data/<split>/*.npy + infos.pkl."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np, yaml
from v2x_prep import pose_matrix, SRC
from ground_map import ground_z
HERE = os.path.dirname(os.path.abspath(__file__)); W = f"{HERE}/work_vru"; OUT = f"{HERE}/ft_data"
STEP = int(os.environ.get("STEP", 2)); MIN_PTS = 3; H_NUSC = 1.84
MAP = {"Pedestrian": "pedestrian", "ScooterRider": "pedestrian", "BicycleRider": "bicycle", "MotorcyleRider": "motorcycle",
       "Car": "car", "Van": "car", "Truck": "truck", "LongVehicle": "truck", "Bus": "bus"}

def sensor_frame(M):
    return np.array([M[0, 3], M[1, 3], float(ground_z(M[0, 3], M[1, 3])) + 0.19 + H_NUSC])

def in_box(p, b):
    c, s = np.cos(b["yaw"]), np.sin(b["yaw"]); dx, dy = p[:, 0] - b["x"], p[:, 1] - b["y"]
    return (np.abs(c * dx + s * dy) <= b["l"] / 2) & (np.abs(-s * dx + c * dy) <= b["w"] / 2) & (np.abs(p[:, 2] - b["z"]) <= b["h"] / 2)

def scenario(path):
    split, scen = path.rstrip("/").split("/")[-2:]; K = f"{split}__{scen}"; od = f"{OUT}/{K}"; os.makedirs(od, exist_ok=True)
    gt = pickle.load(open(f"{W}/{K}/gt_all.pkl", "rb")); ys = sorted(glob.glob(f"{path}/-1/*.yaml")); infos = []
    pos = {(i, b["category_id"], b["track_id"]): (b["x"], b["y"]) for i, v in gt.items() for b in v}   # ids repeat across categories
    for i in range(0, len(ys), STEP):
        for ag in ("-1", "-2"):
            f = ys[i].replace("/-1/", f"/{ag}/")
            if not os.path.exists(f): continue
            y = yaml.load(open(f), Loader=yaml.CSafeLoader); M = pose_matrix(y["lidar_pose"]); off = sensor_frame(M)
            p = np.fromfile(f.replace(".yaml", ".bin"), np.float32).reshape(-1, 4); p = p[np.abs(p[:, :3]).sum(1) > 0.5]
            w = p[:, :3] @ M[:3, :3].T + M[:3, 3]; boxes, names = [], []
            for b in gt[i]:
                if b["type"] not in MAP or in_box(w, b).sum() < MIN_PTS: continue
                a, c = pos.get((i - 1, b["category_id"], b["track_id"])), pos.get((i + 1, b["category_id"], b["track_id"]))
                v = ((np.array(c) - np.array(a)) / 0.2) if a and c else np.zeros(2)
                if np.hypot(*v) > 15: v = np.zeros(2)   # residual id clashes
                boxes.append([b["x"] - off[0], b["y"] - off[1], b["z"] - off[2], b["l"], b["w"], b["h"], b["yaw"], v[0], v[1]]); names.append(MAP[b["type"]])
            pts = np.zeros((len(w), 5), np.float32); pts[:, :3] = w - off; pts[:, 3] = np.clip(p[:, 3], 0, 255)
            fn = f"{od}/{i:06d}{ag}.npy"; np.save(fn, pts)
            infos.append(dict(frame_id=f"{K}/{i:06d}{ag}", lidar_path=fn, gt_boxes=np.array(boxes, np.float32).reshape(-1, 9), gt_names=np.array(names)))
    return infos

if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(f"{SRC}/train*/*/")) if os.path.isdir(p + "-1")]
    with Pool(16) as pool: infos = [x for r in pool.map(scenario, paths) for x in r]
    pickle.dump(infos, open(f"{OUT}/infos_train.pkl", "wb"))
    n = np.concatenate([i["gt_names"] for i in infos]); u, c = np.unique(n, return_counts=True)
    print(len(paths), "scenarios,", len(infos), "samples;", dict(zip(u, c)))
