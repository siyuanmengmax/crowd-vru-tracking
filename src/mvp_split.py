"""Step 1 minimal version: count-guided splitting of merged VRU clusters, applied online before any tracker.
Per frame, for every VRU cluster of the eps0.5 front end, the number of people k is estimated from
  hist  the previous frame's output VRU objects whose centre lies within GATE m of the cluster's points,
  det   the learned detector's VRU boxes (score >= DET_T) whose centre lies within GATE m of the cluster's points,
  max   max(hist, det),
and the cluster is split by k-means on x-y (k reduced until every part keeps >= MIN_PTS points).
The output (tags mvp_hist / mvp_det / mvp_max) uses the eps0.5 layout, so the trackers and eval_vru run unchanged."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from v2x_prep import compute_bbox
from split_prep import owners
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = f"{HERE}/work_vru"
GATE, DET_T, MIN_PTS = 0.5, float(os.environ.get("DET_T", 0.3)), 3
MODES = os.environ.get("MODES", "hist,det,max").split(","); SRC_TAG = os.environ.get("SRC_TAG", "eps0.5"); PFX = os.environ.get("PFX", "mvp"); VRU_CLS = {"pedestrian", "bicycle", "motorcycle"}

MIN_SEP = 0.35  # two people's centres cannot be closer than this (m)

def assign_counts(pts, cids, centres):
    """Each centre counts once, for the VRU cluster whose nearest point is closest (within GATE)."""
    cnt = {}
    if not len(centres) or not len(pts): return cnt
    d, i = cKDTree(pts[:, :2]).query(centres, distance_upper_bound=GATE)
    for dd, ii in zip(d, i):
        if np.isfinite(dd): cnt[cids[ii]] = cnt.get(cids[ii], 0) + 1
    return cnt

def split(p, k):
    while k > 1:
        km = KMeans(k, n_init=3, random_state=0).fit(p[:, :2]); lab = km.labels_; c = km.cluster_centers_
        sep = np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1))[np.triu_indices(k, 1)].min()
        if np.bincount(lab, minlength=k).min() >= MIN_PTS and sep >= MIN_SEP: return [lab == j for j in range(k)]
        k -= 1
    return [np.ones(len(p), bool)]

def scenario(key):
    gt = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb"))
    src = pd.read_csv(f"{WORK}/{key}/{SRC_TAG}/detection_bbox_results.csv")
    det = pd.read_csv(f"{WORK}/{key}/det_voxelnext.csv"); det = det[det.cls.isin(VRU_CLS) & (det.score >= DET_T)]
    D = {f: g[["x", "y"]].to_numpy() for f, g in det.groupby("frame_id")}
    stats = []
    for mode in MODES:
        tag = f"{PFX}_{mode}"; os.makedirs(f"{WORK}/{key}/{tag}/detection", exist_ok=True)
        rows, mem = [], []; prev = np.zeros((0, 2))
        for f in sorted(glob.glob(f"{WORK}/{key}/{SRC_TAG}/detection/*.bin")):
            name = os.path.basename(f); fid = int(name[:6]); boxes = gt[fid]
            pts = np.fromfile(f, np.float32).reshape(-1, 10).astype(np.float64)
            lab0 = src[src.frame_id == fid].set_index("cluster_id")["label"]
            own = owners(pts, boxes) if len(pts) else np.zeros(0, int)
            outs, cid, cur = [], 0, []
            vm = np.isin(pts[:, 9], [c for c, l in lab0.items() if l == 0]) if len(pts) else np.zeros(0, bool)
            KH = assign_counts(pts[vm], pts[vm, 9].astype(int), prev); KD = assign_counts(pts[vm], pts[vm, 9].astype(int), D.get(fid, np.zeros((0, 2))))
            for c in (np.unique(pts[:, 9]).astype(int) if len(pts) else []):
                sel = pts[:, 9] == c; p = pts[sel]; o = own[sel]; vru = lab0.get(c, 3) == 0
                k = 1
                if vru:
                    kh, kd = KH.get(c, 0), KD.get(c, 0)
                    k = max(1, {"hist": kh, "det": kd, "max": max(kh, kd)}[mode])
                parts = split(p, k) if k > 1 else [np.ones(len(p), bool)]
                if vru: stats.append(dict(mode=mode, k_est=k, k_out=len(parts),
                                          k_true=sum(1 for j in np.unique(o[o >= 0]) if (o == j).sum() >= 3 and boxes[j]["category_id"] == 0)))
                for m in parts:
                    q = p[m]; oq = o[m]; ok = oq[oq >= 0]
                    if len(q) < MIN_PTS: continue
                    bb = compute_bbox(q[:, :3])
                    rows.append(dict(frame_id=fid, frame_name=name, cluster_id=cid, label=int(lab0.get(c, 3)), confidence=1.0,
                                     x=bb[0], y=bb[1], z=bb[2], w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=0.5,
                                     avg_reliability=float(q[:, 7].mean()), num_points=len(q)))
                    mem.append(dict(frame_id=fid, cluster_id=cid, owners={boxes[j]["track_id"]: int((ok == j).sum()) for j in np.unique(ok)}))
                    qq = q.copy(); qq[:, 9] = cid; outs.append(qq); cid += 1
                    if vru: cur.append(q[:, :2].mean(0))
            prev = np.array(cur).reshape(-1, 2)
            (np.vstack(outs) if outs else np.zeros((0, 10))).astype(np.float32).tofile(f"{WORK}/{key}/{tag}/detection/{name}")
        pd.DataFrame(rows).to_csv(f"{WORK}/{key}/{tag}/detection_bbox_results.csv", index=False)
        pickle.dump(mem, open(f"{WORK}/{key}/{tag}/members.pkl", "wb"))
    return stats

if __name__ == "__main__":
    keys = sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/*__*"))
    if os.environ.get("LIMIT"): keys = keys[:int(os.environ["LIMIT"])]
    with Pool(int(os.environ.get("NPROC", 20))) as pool: st = pd.DataFrame([s for ss in pool.map(scenario, keys) for s in ss])
    st.to_csv(f"{HERE}/{PFX}_count_stats.csv", index=False)
    for mode, g in st.groupby("mode"):
        print(pd.crosstab(g.k_true.clip(0, 4), g.k_out.clip(0, 4), normalize="index").round(2))
        m = g[g.k_true >= 1]
        print(mode, "count accuracy", round((m.k_out == m.k_true).mean(), 3), "| under", round((m.k_out < m.k_true).mean(), 3),
              "| over", round((m.k_out > m.k_true).mean(), 3), "| merged (k_true>=2) correct", round((m[m.k_true >= 2].k_out == m[m.k_true >= 2].k_true).mean(), 3))
