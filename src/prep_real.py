"""Step 2b front end: real background removal instead of oracle foreground.
Foreground = returns whose 0.2 m voxel (and its 26 neighbours) is occupied in < THETA of the background frames
(bg_build.py); both LiDARs are merged in the world frame and clustered with DBSCAN(eps 0.5, min_samples 3).
Writes tag real0.5 with label = annotated majority class of the cluster's returns (3 if most returns are outside
every cuboid), so the effect of the foreground alone can be read; relabel_size.py derives a label-free variant.
Membership (which GT objects own the returns) is kept for evaluation only."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, yaml
from sklearn.cluster import DBSCAN
from v2x_prep import pose_matrix, calculate_reliability, compute_bbox, SRC, MARGIN, GROUND_CLEAR
from bg_build import keys
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru"); TAG = "real0.5"
BG_DIR = os.environ.get("BG_DIR", HERE)   # bg_-1.npz, bg_-2.npz from bg_build.py
THETA, EPS = 0.2, 0.5
BG = {ag: dict(np.load(f"{BG_DIR}/bg_{ag}.npz")) for ag in ("-1", "-2")}  # eager: forked workers must not share the zip handle
OFF = [(dx << 34) + (dy << 17) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

def occ(ag, p):
    k0 = keys(p); b = BG[ag]; out = np.zeros(len(p))
    for o in OFF:
        k = k0 + o; i = np.clip(np.searchsorted(b["keys"], k), 0, len(b["keys"]) - 1)
        out = np.maximum(out, np.where(b["keys"][i] == k, b["occ"][i], 0.0))
    return out

def scenario(path):
    key = "__".join(path.rstrip("/").split("/")[-2:]); os.makedirs(f"{WORK}/{key}/{TAG}/detection", exist_ok=True)
    gt = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb"))
    rows, mem = [], []
    for i, yf in enumerate(sorted(glob.glob(f"{path}/-1/*.yaml"))):
        clouds, objs = [], {}
        for lid, ag in enumerate(["-1", "-2"]):
            f = yf.replace("/-1/", f"/{ag}/")
            if not os.path.exists(f): continue
            y = yaml.load(open(f), Loader=yaml.CSafeLoader)
            p = np.fromfile(f.replace(".yaml", ".bin"), np.float32).reshape(-1, 4); p = p[np.abs(p[:, :3]).sum(1) > 0]
            p = p[occ(ag, p) < THETA]
            r = np.linalg.norm(p[:, :3], axis=1); refl = np.clip(p[:, 3], 0, 255); M = pose_matrix(y["lidar_pose"])
            clouds.append(np.column_stack([p[:, :3] @ M[:3, :3].T + M[:3, 3], r, refl, np.zeros(len(p)), np.ones(len(p)),
                                           calculate_reliability(r.astype(np.float32), refl.astype(np.float32)), np.full(len(p), lid)]))
            objs.update(y.get("vehicles") or {})
        fg = np.vstack(clouds); boxes = gt[i]
        # ownership with the full cuboid rotation; gt_all.pkl lists the boxes in the same objs order (prep_v2x_vru)
        own = np.full(len(fg), -1); best = np.full(len(fg), np.inf)
        for j, v in enumerate(objs.values()):
            Mv = pose_matrix(list(v["location"]) + list(v["angle"])); c = (Mv @ np.r_[v.get("center", [0, 0, 0]), 1])[:3]
            assert abs(boxes[j]["x"] - c[0]) < 1e-6
            ex = np.array(v["extent"]); q = (fg[:, :3] - c) @ Mv[:3, :3]
            m = (np.abs(q[:, 0]) < ex[0] + MARGIN) & (np.abs(q[:, 1]) < ex[1] + MARGIN) & (q[:, 2] > -ex[2] + GROUND_CLEAR) & (q[:, 2] < ex[2] + MARGIN)
            d = np.hypot(q[:, 0], q[:, 1]); t = m & (d < best); own[t] = j; best[t] = d[t]
        lab = DBSCAN(eps=EPS, min_samples=3).fit_predict(fg[:, :3]) if len(fg) else np.zeros(0, int)
        out, cid, name = [], 0, f"{i:06d}.bin"
        for c in np.unique(lab[lab >= 0]):
            sel = lab == c; p = fg[sel]; o = own[sel]; ok = o[o >= 0]
            if len(p) < 3: continue
            cats = np.bincount([boxes[k]["category_id"] for k in ok], minlength=4)
            label = int(np.argmax(cats)) if len(ok) >= 0.5 * len(p) else 3
            bb = compute_bbox(p[:, :3])
            rows.append(dict(frame_id=i, frame_name=name, cluster_id=cid, label=label, confidence=1.0, x=bb[0], y=bb[1], z=bb[2],
                             w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=EPS, avg_reliability=float(p[:, 7].mean()),
                             num_points=len(p), zmin=float(p[:, 2].min()), zmax=float(p[:, 2].max()), owned=len(ok) / len(p)))
            mem.append(dict(frame_id=i, cluster_id=cid, owners={boxes[k]["track_id"]: int((ok == k).sum()) for k in np.unique(ok)}))
            out.append(np.column_stack([p, np.full(len(p), cid)])); cid += 1
        (np.vstack(out) if out else np.zeros((0, 10))).astype(np.float32).tofile(f"{WORK}/{key}/{TAG}/detection/{name}")
    pd.DataFrame(rows).to_csv(f"{WORK}/{key}/{TAG}/detection_bbox_results.csv", index=False)
    pickle.dump(mem, open(f"{WORK}/{key}/{TAG}/members.pkl", "wb"))
    return key

if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob(f"{SRC}/*/*/")) if os.path.isdir(p + "-1")]
    if os.environ.get("LIMIT"): paths = paths[:int(os.environ["LIMIT"])]
    if os.environ.get("ONLY"): paths = [p for p in paths if "__".join(p.rstrip("/").split("/")[-2:]) in os.environ["ONLY"].split(",")]
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        for k in pool.imap_unordered(scenario, paths): print(k, flush=True)
