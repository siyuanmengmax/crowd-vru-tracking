"""VRU pre-check front end on V2X-Real infrastructure LiDARs (-1, -2).
Foreground = returns inside ANY annotated cuboid (oracle background removal); clusters come from DBSCAN on the
foreground, so nearby road users can merge and one object can split, as in a real unsupervised front end.
Cluster class = majority annotated class of its points (0 VRU, 1 car/van, 2 truck/bus, 3 other).
Writes the same layout as tumtraf/v2x_prep.py so the existing trackers run unchanged, plus gt_all.pkl and
per-cluster membership (which GT objects own its points) for merge/split analysis."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, yaml
from sklearn.cluster import DBSCAN
from v2x_prep import pose_matrix, calculate_reliability, compute_bbox, SRC, MARGIN, GROUND_CLEAR
OUT = os.environ.get("WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_vru"))
EPS = [float(e) for e in os.environ.get("EPS", "0.5,0.8").split(",")]
VRU = {"Pedestrian", "ScooterRider", "BicycleRider", "MotorcyleRider"}
def cat(t): return 0 if t in VRU else 1 if t in ("Car", "Van") else 2 if t in ("Truck", "Bus", "LongVehicle") else 3

def scenario(path):
    key = "__".join(path.rstrip("/").split("/")[-2:])
    for e in EPS: os.makedirs(f"{OUT}/{key}/eps{e}/detection", exist_ok=True)
    frames = sorted(glob.glob(f"{path}/-1/*.yaml"))
    ids, gt = {}, {}; rows = {e: [] for e in EPS}; members = {e: [] for e in EPS}
    for i, yf in enumerate(frames):
        clouds, objs = [], {}
        for lid, ag in enumerate(["-1", "-2"]):
            f = yf.replace("/-1/", f"/{ag}/")
            if not os.path.exists(f): continue
            y = yaml.load(open(f), Loader=yaml.CSafeLoader)
            p = np.fromfile(f.replace(".yaml", ".bin"), np.float32).reshape(-1, 4); p = p[np.abs(p[:, :3]).sum(1) > 0]
            r = np.linalg.norm(p[:, :3], axis=1); refl = np.clip(p[:, 3], 0, 255); M = pose_matrix(y["lidar_pose"])
            clouds.append(np.column_stack([p[:, :3] @ M[:3, :3].T + M[:3, 3], r, refl, np.zeros(len(p)), np.ones(len(p)),
                                           calculate_reliability(r.astype(np.float32), refl.astype(np.float32)), np.full(len(p), lid)]))
            objs.update(y.get("vehicles") or {})
        pts = np.vstack(clouds); owner = np.full(len(pts), -1); best = np.full(len(pts), np.inf); boxes = []
        for vid, v in objs.items():
            Mv = pose_matrix(list(v["location"]) + list(v["angle"])); c = (Mv @ np.r_[v.get("center", [0, 0, 0]), 1])[:3]
            ex = np.array(v["extent"]); q = (pts[:, :3] - c) @ Mv[:3, :3]
            m = (np.abs(q[:, 0]) < ex[0] + MARGIN) & (np.abs(q[:, 1]) < ex[1] + MARGIN) & (q[:, 2] > -ex[2] + GROUND_CLEAR) & (q[:, 2] < ex[2] + MARGIN)
            d = np.hypot(q[:, 0], q[:, 1]); take = m & (d < best); owner[take] = len(boxes); best[take] = d[take]
            boxes.append(dict(x=c[0], y=c[1], z=c[2], l=2 * ex[0], w=2 * ex[1], h=2 * ex[2], yaw=float(np.radians(v["angle"][1])),
                              track_id=ids.setdefault(vid, len(ids)), category_id=cat(v.get("obj_type", "")), type=v.get("obj_type", ""),
                              npts=0))
        for j in range(len(boxes)): boxes[j]["npts"] = int((owner == j).sum())
        gt[i] = boxes
        fg = pts[owner >= 0]; fo = owner[owner >= 0]; name = f"{i:06d}.bin"
        for e in EPS:
            lab = DBSCAN(eps=e, min_samples=3).fit_predict(fg[:, :3]) if len(fg) else np.zeros(0, int)
            out = []; cid = 0
            for c in np.unique(lab[lab >= 0]):
                sel = lab == c; p = fg[sel]; own = fo[sel]
                if len(p) < 3: continue
                cnt = np.bincount(own, minlength=len(boxes)); cats = np.bincount([boxes[k]["category_id"] for k in own], minlength=4)
                bb = compute_bbox(p[:, :3])
                rows[e].append(dict(frame_id=i, frame_name=name, cluster_id=cid, label=int(np.argmax(cats)), confidence=1.0,
                                    x=bb[0], y=bb[1], z=bb[2], w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=e,
                                    avg_reliability=float(p[:, 7].mean()), num_points=len(p)))
                members[e].append(dict(frame_id=i, cluster_id=cid, owners={boxes[k]["track_id"]: int(cnt[k]) for k in np.nonzero(cnt)[0]}))
                out.append(np.column_stack([p, np.full(len(p), cid)])); cid += 1
            (np.vstack(out) if out else np.zeros((0, 10))).astype(np.float32).tofile(f"{OUT}/{key}/eps{e}/detection/{name}")
    pickle.dump(gt, open(f"{OUT}/{key}/gt_all.pkl", "wb"))
    for e in EPS:
        pd.DataFrame(rows[e]).to_csv(f"{OUT}/{key}/eps{e}/detection_bbox_results.csv", index=False)
        pickle.dump(members[e], open(f"{OUT}/{key}/eps{e}/members.pkl", "wb"))
    return dict(scenario=key, frames=len(frames), vru_tracks=len({b["track_id"] for v in gt.values() for b in v if b["category_id"] == 0}))

if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(f"{SRC}/*/*/")) if os.path.isdir(p + "-1")]
    if os.environ.get("LIMIT"): paths = paths[:int(os.environ["LIMIT"])]
    if os.environ.get("ONLY"): paths = [p for p in paths if "__".join(p.rstrip("/").split("/")[-2:]) in os.environ["ONLY"].split(",")]
    print(len(paths), "scenarios", flush=True)
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        res = pd.DataFrame(pool.map(scenario, paths, chunksize=1))
    res.to_csv(f"{OUT}/scenarios.csv", index=False); print(res.sum(numeric_only=True).to_dict())
