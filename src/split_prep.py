"""Upper bounds for splitting merged clusters (pre-check A).
Starts from the eps0.5 DBSCAN clusters and writes two new tags in the same layout:
  oracle0.5  each cluster is split by the annotated object that owns its points (perfect splitting);
  count0.5   a cluster owning >= 2 objects (>= 3 points each) is split by k-means on x-y with k = true count
             (what a geometric splitter achieves when the number of people is known).
Point ownership is recomputed from gt_all.pkl with yaw-only cuboids; points outside every cuboid keep the
cluster's majority owner."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from v2x_prep import compute_bbox, MARGIN
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = f"{HERE}/work_vru"; SRC_TAG = "eps0.5"
TAGS = ["oracle0.5", "count0.5"]

def owners(p, boxes):
    own = np.full(len(p), -1); best = np.full(len(p), np.inf)
    for j, b in enumerate(boxes):
        c, s = np.cos(b["yaw"]), np.sin(b["yaw"]); dx, dy = p[:, 0] - b["x"], p[:, 1] - b["y"]
        u, v = c * dx + s * dy, -s * dx + c * dy
        m = (np.abs(u) < b["l"] / 2 + MARGIN) & (np.abs(v) < b["w"] / 2 + MARGIN) & (np.abs(p[:, 2] - b["z"]) < b["h"] / 2 + MARGIN)
        d = np.hypot(u, v); t = m & (d < best); own[t] = j; best[t] = d[t]
    return own

def scenario(key):
    gt = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb"))
    rows = {t: [] for t in TAGS}; mem = {t: [] for t in TAGS}
    for t in TAGS: os.makedirs(f"{WORK}/{key}/{t}/detection", exist_ok=True)
    for f in sorted(glob.glob(f"{WORK}/{key}/{SRC_TAG}/detection/*.bin")):
        name = os.path.basename(f); fid = int(name[:6]); boxes = gt[fid]
        pts = np.fromfile(f, np.float32).reshape(-1, 10).astype(np.float64)
        own = owners(pts, boxes) if len(pts) else np.zeros(0, int)
        outs = {t: [] for t in TAGS}; cid = {t: 0 for t in TAGS}
        for c in np.unique(pts[:, 9]).astype(int) if len(pts) else []:
            sel = pts[:, 9] == c; p = pts[sel]; o = own[sel].copy()
            if (o >= 0).any(): o[o < 0] = np.bincount(o[o >= 0]).argmax()
            else: o[:] = -1
            cnt = {k: int((o == k).sum()) for k in np.unique(o) if k >= 0}
            real = [k for k, n in cnt.items() if n >= 3]
            parts = {"oracle0.5": [o == k for k in np.unique(o)]}
            if len(real) >= 2:
                lab = KMeans(len(real), n_init=3, random_state=0).fit_predict(p[:, :2])
                parts["count0.5"] = [lab == k for k in range(len(real))]
            else:
                parts["count0.5"] = [np.ones(len(p), bool)]
            for t in TAGS:
                for m in parts[t]:
                    q = p[m]; oq = o[m]
                    if len(q) < 3: continue
                    ok = oq[oq >= 0]; cats = np.bincount([boxes[k]["category_id"] for k in ok], minlength=4) if len(ok) else np.array([0, 0, 0, 1])
                    bb = compute_bbox(q[:, :3])
                    rows[t].append(dict(frame_id=fid, frame_name=name, cluster_id=cid[t], label=int(np.argmax(cats)), confidence=1.0,
                                        x=bb[0], y=bb[1], z=bb[2], w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=0.5,
                                        avg_reliability=float(q[:, 7].mean()), num_points=len(q)))
                    mem[t].append(dict(frame_id=fid, cluster_id=cid[t],
                                       owners={boxes[k]["track_id"]: int((ok == k).sum()) for k in np.unique(ok)}))
                    qq = q.copy(); qq[:, 9] = cid[t]; outs[t].append(qq); cid[t] += 1
        for t in TAGS:
            (np.vstack(outs[t]) if outs[t] else np.zeros((0, 10))).astype(np.float32).tofile(f"{WORK}/{key}/{t}/detection/{name}")
    for t in TAGS:
        pd.DataFrame(rows[t]).to_csv(f"{WORK}/{key}/{t}/detection_bbox_results.csv", index=False)
        pickle.dump(mem[t], open(f"{WORK}/{key}/{t}/members.pkl", "wb"))
    return key

if __name__ == "__main__":
    keys = sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/*__*"))
    if os.environ.get("LIMIT"): keys = keys[:int(os.environ["LIMIT"])]
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for k in pool.imap_unordered(scenario, keys): print(k, flush=True)
