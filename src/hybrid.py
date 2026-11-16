"""Hybrid v1: label-free cluster front end + learned detector, combined online before an unchanged tracker.
Input per frame: real-foreground DBSCAN clusters (real0.5) and VoxelNeXt boxes (det_voxelnext.csv).
A light object memory (previous frame's outputs, nearest within GATE) carries, per object, the frame of its last
detector VRU confirmation, its number of confirmations, and whether the detector saw several people in it.
  candidate  cluster geometry can hold people: vertical extent 0.7-2.4 m, footprint <= 3.0 x 1.8 m, >= 5 returns;
  count k    k_hist (memory objects falling in the cluster); raised to k_det (detector VRU boxes in the cluster)
             only if the detector also saw several people there in the previous frame and the footprint is >= 0.8 m;
  split      k-means on x-y (parts >= 3 returns, centres >= 0.35 m apart);
  class      a part is VRU if a detector VRU box confirms it now, or its object has >= MIN_HITS confirmations and the
             last one is at most MEM frames old; a part a detector vehicle box covers is never VRU.
Non-VRU clusters keep the label-free geometric class (relabel_size.label). Writes tag hyb_<name>.
v2 options: RELAX=1 also takes low clusters (vertical extent >= 0.25 m, footprint <= 1.5 m, partly occluded people)
as candidates when a detector VRU box hits them; COAST=n keeps an unassociated memory object for n frames so a
short break in the cluster sequence does not reset its confirmations; MEM large = confirmations never expire.
Ablation: SPLIT=0 never splits candidate clusters; DET_COUNT=0 takes k from memory only; MEM=0 = no memory confirmation.
Count fix (2026-11-11, after the crosswalk diagnosis): DEDUP=r counts detector VRU boxes (any VRU class) whose centres
are closer than r m as one person (greedy, highest score kept) for k_det and the multi flag; confirmation is unchanged.
Baseline: KDE_TAU>0 replaces count and split of candidate clusters by single-frame KDE density peaks (after Wang et al. 2022)."""
import glob, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from bbox_fast import compute_bbox
from mvp_split import split, assign_counts
from relabel_size import label as geo_label
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru")
P = dict(DET_T=0.3, GATE=0.6, MIN_HITS=2, MEM=30, DET_COUNT=1, RELAX=0, COAST=0, SPLIT=1, KDE_TAU=0.0, DEDUP=0.0, NAME="v1")
P.update({k: type(v)(os.environ[k]) for k, v in P.items() if k in os.environ})
TIME = int(os.environ.get("TIME", 0))   # TIME=1 (runtime.sh): per-frame time without GT ownership and file I/O -> <tag>/frame_times.npy
DETF = os.environ.get("DETF", "det_voxelnext")   # detector boxes; DETF=det_ft_voxelnext NAME=v1ft = hybrid with the V2X-Real fine-tuned detector
VRU_CLS = {"pedestrian", "bicycle", "motorcycle"}; VEH_CLS = {"car", "truck", "bus", "trailer", "construction_vehicle"}

def candidate(r):
    lo, sh, H = max(r.w, r.l), min(r.w, r.l), r.zmax - r.zmin
    return 0.7 <= H <= 2.4 and lo <= 3.0 and sh <= 1.8 and r.num_points >= 5

def relaxed(r, p, dv):
    if not len(dv): return False
    H, lo = r.zmax - r.zmin, max(r.w, r.l)
    return H >= 0.25 and lo <= 1.5 and np.isfinite(cKDTree(p[:, :2]).query(dv, distance_upper_bound=0.5)[0]).any()

def kde_split(p, tau, step=0.05, min_pts=3, min_sep=0.35):
    """Baseline after Wang et al. (2022): count = local maxima of a Gaussian KDE (window tau) of the x-y points,
    points go to the nearest peak. Peaks closer than min_sep are suppressed (weaker one dropped); peaks keeping
    < min_pts points are dropped and their points reassigned."""
    from scipy.ndimage import maximum_filter
    xy = p[:, :2]; lo = xy.min(0) - 2 * tau; hi = xy.max(0) + 2 * tau
    gx = np.arange(lo[0], hi[0] + step, step); gy = np.arange(lo[1], hi[1] + step, step)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    dens = np.zeros(len(G))
    for i in range(0, len(xy), 512):
        d2 = ((G[:, None, :] - xy[None, i:i + 512]) ** 2).sum(-1); dens += np.exp(-d2 / (2 * tau ** 2)).sum(1)
    D = dens.reshape(len(gx), len(gy)); pk = np.argwhere((D == maximum_filter(D, size=3, mode="constant")) & (D > 0))
    pk = pk[np.argsort(-D[pk[:, 0], pk[:, 1]])]; C = []
    for i, j in pk:
        c = np.array([gx[i], gy[j]])
        if all(np.hypot(*(c - k)) >= min_sep for k in C): C.append(c)
    C = np.array(C)
    while len(C) > 1:
        lab = cKDTree(C).query(xy)[1]; n = np.bincount(lab, minlength=len(C))
        if n.min() >= min_pts: return [lab == j for j in range(len(C))]
        C = np.delete(C, n.argmin(), 0)
    return [np.ones(len(p), bool)]

def dedup(g, r):
    """Greedy suppression of VRU boxes with centres closer than r m (any VRU class), highest score first."""
    g = g.sort_values("score", ascending=False); xy = g[["x", "y"]].to_numpy(); keep = []
    for i in range(len(xy)):
        if all(np.hypot(*(xy[i] - xy[j])) >= r for j in keep): keep.append(i)
    return xy[keep]

def covered(q, boxes):
    for b in boxes:
        c, s = np.cos(b[3]), np.sin(b[3]); dx, dy = q[:, 0] - b[0], q[:, 1] - b[1]
        if ((np.abs(c * dx + s * dy) < b[4] / 2) & (np.abs(-s * dx + c * dy) < b[5] / 2)).mean() > 0.5: return True
    return False

def scenario(key):
    gt = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb"))
    src = pd.read_csv(f"{WORK}/{key}/real0.5/detection_bbox_results.csv"); src["glab"] = geo_label(src)
    det = pd.read_csv(f"{WORK}/{key}/{DETF}.csv"); det = det[det.score >= P["DET_T"]]
    DV = {f: g[["x", "y"]].to_numpy() for f, g in det[det.cls.isin(VRU_CLS)].groupby("frame_id")}
    DN = {f: dedup(g, P["DEDUP"]) for f, g in det[det.cls.isin(VRU_CLS)].groupby("frame_id")} if P["DEDUP"] > 0 else DV
    DC = {f: g[["x", "y", "z", "yaw", "l", "w"]].to_numpy() for f, g in det[det.cls.isin(VEH_CLS)].groupby("frame_id")}
    tag = f"hyb_{P['NAME']}"; os.makedirs(f"{WORK}/{key}/{tag}/detection", exist_ok=True)
    rows, mem, stats, times = [], [], [], []
    prev = np.zeros((0, 2)); prev_state = []      # state: (last_hit_frame, hits, multi_flag)
    memxy = np.zeros((0, 2)); mem_state = []       # memory = this frame's objects + coasting ones; state + (age,)
    from split_prep import owners
    for f in sorted(glob.glob(f"{WORK}/{key}/real0.5/detection/*.bin")):
        name = os.path.basename(f); fid = int(name[:6]); boxes = gt[fid]
        pts = np.fromfile(f, np.float32).reshape(-1, 10).astype(np.float64)
        t0 = time.perf_counter()
        S = src[src.frame_id == fid].set_index("cluster_id")
        own = np.full(len(pts), -1) if TIME else owners(pts, boxes) if len(pts) else np.zeros(0, int)
        dv = DV.get(fid, np.zeros((0, 2))); dn = DN.get(fid, np.zeros((0, 2))); dc = DC.get(fid, np.zeros((0, 6)))
        cand = [c for c in S.index if candidate(S.loc[c]) or (P["RELAX"] and relaxed(S.loc[c], pts[pts[:, 9] == c], dv))]
        cm = np.isin(pts[:, 9], cand) if len(pts) else np.zeros(0, bool)
        KH = assign_counts(pts[cm], pts[cm, 9].astype(int), prev); KD = assign_counts(pts[cm], pts[cm, 9].astype(int), dn)
        ptree = cKDTree(prev) if len(prev) else None; mtree = cKDTree(memxy) if len(memxy) else None; used = set()
        outs, cid, cur, cur_state = [], 0, [], []
        for c in (np.unique(pts[:, 9]).astype(int) if len(pts) else []):
            sel = pts[:, 9] == c; p = pts[sel]; o = own[sel]; r = S.loc[c]
            if c in cand:
                k = max(1, KH.get(c, 0)); kd = KD.get(c, 0)
                if P["DET_COUNT"] and kd > k and max(r.w, r.l) >= 0.8 and ptree is not None:
                    _, j = ptree.query(p[:, :2].mean(0), distance_upper_bound=P["GATE"])
                    if j < len(prev) and prev_state[j][2]: k = kd
                if P["KDE_TAU"] > 0: parts = kde_split(p, P["KDE_TAU"])   # baseline: count and split from KDE peaks only
                else: parts = split(p, k) if k > 1 and P["SPLIT"] else [np.ones(len(p), bool)]
            else:
                parts = [np.ones(len(p), bool)]
            for m in parts:
                q = p[m]; oq = o[m]; ok = oq[oq >= 0]
                if len(q) < 3: continue
                lab = int(r.glab)
                if c in cand:
                    ctr = q[:, :2].mean(0)
                    last, hits = -10 ** 6, 0
                    if mtree is not None:
                        d, j = mtree.query(ctr, distance_upper_bound=P["GATE"])
                        if j < len(memxy): last, hits = mem_state[j][0], mem_state[j][1]; used.add(j)
                    hit = len(dv) and np.isfinite(cKDTree(q[:, :2]).query(dv, distance_upper_bound=0.5)[0]).any()
                    if hit: last, hits = fid, hits + 1
                    veh = len(dc) and covered(q, dc)
                    is_vru = (not veh) and (hit or (hits >= P["MIN_HITS"] and fid - last <= P["MEM"]))
                    if is_vru: lab = 0
                    elif lab == 0: lab = 3
                    cur.append(ctr); cur_state.append((last, hits, KD.get(c, 0) >= 2))
                elif lab == 0: lab = 3
                bb = compute_bbox(q[:, :3])
                rows.append(dict(frame_id=fid, frame_name=name, cluster_id=cid, label=lab, confidence=1.0, x=bb[0], y=bb[1], z=bb[2],
                                 w=bb[3], l=bb[4], h=bb[5], yaw=bb[6], avg_eps=0.5, avg_reliability=float(q[:, 7].mean()), num_points=len(q)))
                mem.append(dict(frame_id=fid, cluster_id=cid, owners={boxes[j]["track_id"]: int((ok == j).sum()) for j in np.unique(ok)}))
                qq = q.copy(); qq[:, 9] = cid; outs.append(qq); cid += 1
        prev = np.array(cur).reshape(-1, 2); prev_state = cur_state
        keep = [j for j in range(len(memxy)) if j not in used and mem_state[j][3] < P["COAST"]]
        memxy = np.vstack([prev, memxy[keep]]) if keep else prev
        mem_state = [s + (0,) for s in cur_state] + [mem_state[j][:3] + (mem_state[j][3] + 1,) for j in keep]
        times.append(time.perf_counter() - t0)
        (np.vstack(outs) if outs else np.zeros((0, 10))).astype(np.float32).tofile(f"{WORK}/{key}/{tag}/detection/{name}")
    pd.DataFrame(rows).to_csv(f"{WORK}/{key}/{tag}/detection_bbox_results.csv", index=False)
    pickle.dump(mem, open(f"{WORK}/{key}/{tag}/members.pkl", "wb"))
    if TIME: np.save(f"{WORK}/{key}/{tag}/frame_times.npy", np.array(times))
    return key

if __name__ == "__main__":
    keys = sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/*__*") if os.path.exists(f"{p}/{DETF}.csv"))
    if os.environ.get("SPLITS"): keys = [k for k in keys if k.split("__")[0] in os.environ["SPLITS"].split(",")]   # e.g. SPLITS=train1,train2,train3,train4
    if os.environ.get("LIMIT"): keys = keys[:int(os.environ["LIMIT"])]
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        for k in pool.imap_unordered(scenario, keys): print(k, flush=True)
