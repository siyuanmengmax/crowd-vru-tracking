"""V2X-Real, infrastructure LiDARs only (agents -1 and -2): annotation-guided segmentation front end.

Same construction as step2_annot_seg.py: returns of both roadside LiDARs inside each annotated cuboid form
one cluster (>= MIN_POINTS), the box is fitted to the points, cluster IDs are shuffled per frame.
Ground truth: union of the two units' annotations (IDs are shared); Car/Van -> 1, Truck/Bus -> 2.
Intensity is clipped to [0, 255] as a stand-in for reflectivity in the FRGB3D reliability model.
"""
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from frgb3d_utils import calculate_reliability  # noqa: E402
from compute_bbox import compute_bbox  # noqa: E402

SRC = os.environ.get("V2X_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "v2x_real"))
OUT = os.environ.get("V2X_SEG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_v2x"))
MIN_POINTS = int(os.environ.get("MIN_POINTS", 5))
TAG = f"seg{MIN_POINTS}"
CLASS = {"Car": 1, "Van": 1, "Truck": 2, "Bus": 2, "Pedestrian": 0}
MARGIN, GROUND_CLEAR = 0.1, 0.15


def pose_matrix(p):
    x, y, z, roll, yaw, pitch = p
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    M = np.eye(4)
    M[:3, 3] = [x, y, z]
    M[0, :3] = [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr]
    M[1, :3] = [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr]
    M[2, :3] = [sp, -cp * sr, cp * cr]
    return M


def scenario(path):
    key = "__".join(path.rstrip("/").split("/")[-2:])
    os.makedirs(f"{OUT}/{key}/{TAG}/detection", exist_ok=True)
    frames = sorted(glob.glob(f"{path}/-1/*.yaml"))
    ids, gt, rows = {}, {}, []
    for i, yf in enumerate(frames):
        rng = np.random.default_rng(i)
        clouds, objs = [], {}
        for lid, ag in enumerate(["-1", "-2"]):
            f = yf.replace("/-1/", f"/{ag}/")
            if not os.path.exists(f):
                continue
            y = yaml.load(open(f), Loader=yaml.CSafeLoader)
            p = np.fromfile(f.replace(".yaml", ".bin"), np.float32).reshape(-1, 4)
            p = p[np.abs(p[:, :3]).sum(1) > 0]
            r = np.linalg.norm(p[:, :3], axis=1)
            refl = np.clip(p[:, 3], 0, 255)
            M = pose_matrix(y["lidar_pose"])
            clouds.append(np.column_stack([p[:, :3] @ M[:3, :3].T + M[:3, 3], r, refl, np.zeros(len(p)), np.ones(len(p)),
                                           calculate_reliability(r.astype(np.float32), refl.astype(np.float32)),
                                           np.full(len(p), lid)]))
            objs.update(y.get("vehicles") or {})
        pts = np.vstack(clouds)
        owner = np.full(len(pts), -1); best = np.full(len(pts), np.inf)
        boxes = []
        for vid, v in objs.items():
            Mv = pose_matrix(list(v["location"]) + list(v["angle"]))
            c = (Mv @ np.r_[v.get("center", [0, 0, 0]), 1])[:3]
            e = np.array(v["extent"])
            q = (pts[:, :3] - c) @ Mv[:3, :3]
            m = (np.abs(q[:, 0]) < e[0] + MARGIN) & (np.abs(q[:, 1]) < e[1] + MARGIN) & (q[:, 2] > -e[2] + GROUND_CLEAR) & (q[:, 2] < e[2] + MARGIN)
            d = np.hypot(q[:, 0], q[:, 1])
            take = m & (d < best); owner[take] = len(boxes); best[take] = d[take]
            cat = CLASS.get(v.get("obj_type", ""), 3)
            boxes.append(dict(x=c[0], y=c[1], z=c[2], l=2 * e[0], w=2 * e[1], h=2 * e[2], yaw=float(np.radians(v["angle"][1])),
                              track_id=ids.setdefault(vid, len(ids)), category_id=cat, type=v.get("obj_type", "")))
        gt[i] = [b for b in boxes if b["category_id"] in (1, 2)]
        keep = [j for j in range(len(boxes)) if (owner == j).sum() >= MIN_POINTS]
        out = []
        name = f"{i:06d}.bin"
        for cid, k in enumerate(rng.permutation(len(keep))):
            j = keep[k]; p = pts[owner == j]
            bb = compute_bbox(p[:, :3])
            rows.append(dict(frame_id=i, frame_name=name, cluster_id=cid, label=boxes[j]["category_id"], confidence=1.0,
                             x=bb[0], y=bb[1], z=bb[2], w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=0.0,
                             avg_reliability=float(p[:, 7].mean()), num_points=len(p)))
            out.append(np.column_stack([p, np.full(len(p), cid)]))
        (np.vstack(out) if out else np.zeros((0, 10))).astype(np.float32).tofile(f"{OUT}/{key}/{TAG}/detection/{name}")
    pickle.dump(gt, open(f"{OUT}/{key}/gt.pkl", "wb"))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/{key}/{TAG}/detection_bbox_results.csv", index=False)
    return dict(scenario=key, frames=len(frames), gt_tracks=len({b["track_id"] for v in gt.values() for b in v}),
                gt_boxes=sum(len(v) for v in gt.values()), veh_det=int(df.label.isin([1, 2]).sum()) if len(df) else 0,
                median_pts=float(df[df.label.isin([1, 2])].num_points.median()) if len(df) else 0)


if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(f"{SRC}/*/*/")) if os.path.isdir(p + "-1")
             and not os.path.exists(f"{OUT}/{'__'.join(p.rstrip('/').split('/')[-2:])}/gt.pkl")]  # new scenarios only
    print(len(paths), "new scenarios")
    with Pool(24) as pool:
        res = pd.DataFrame(pool.map(scenario, paths, chunksize=1))
    res.to_csv(f"{OUT}/scenarios_{TAG}_added.csv", index=False)
    print(res.to_string(index=False))
    print(res[["frames", "gt_tracks", "gt_boxes", "veh_det"]].sum().to_dict())
