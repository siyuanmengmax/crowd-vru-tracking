"""Evaluate VRU tracking (pedestrians + riders) and attribute ID switches to crowding and cluster merges.
Matching: BEV centre distance <= MATCH m, Hungarian, per frame. GT = VRU boxes with >= 3 foreground returns.
Predictions = tracker boxes whose final label is 0 (VRU). IDS = standard (change of predicted ID w.r.t. the last match)."""
import os, sys, glob, pickle
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru")
ANNOT = os.environ.get("ANNOT_DIR", os.path.join(HERE, "..", "annotations"))   # ignore-region review (Zenodo release)
RESULTS = os.environ.get("RESULTS", os.path.join(HERE, "..", "results")); os.makedirs(RESULTS, exist_ok=True)
MATCH = float(os.environ.get("MATCH", 1.0))
FILES = {"AB3DMOT": "AB3DMOT_matched.csv", "SinglePoint": "SinglePointUKF_matched.csv", "Immortal": "Immortal_matched.csv"}
BINS = [0, 0.8, 1.5, 3.0, np.inf]; BLAB = ["<0.8", "0.8-1.5", "1.5-3", ">=3"]

def one(job):
    key, tag = job
    gt_all = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb")); mem = pickle.load(open(f"{WORK}/{key}/{tag}/members.pkl", "rb"))
    # merge status per (frame, gt id): the cluster holding most of its points also holds >= 3 points of another object
    merged = set(); main = {}
    for m in mem:
        for k, n in m["owners"].items():
            if n > main.get((m["frame_id"], k), (0, None))[0]: main[(m["frame_id"], k)] = (n, m)
    for (f, k), (n, m) in main.items():
        if any(o != k and c >= 3 for o, c in m["owners"].items()): merged.add((f, k))
    G = {}; nn = {}
    for f, bs in gt_all.items():
        v = [b for b in bs if b["category_id"] == 0 and b["npts"] >= 3]; G[f] = v
        allv = [b for b in bs if b["category_id"] == 0]
        for b in v:
            d = [np.hypot(b["x"] - o["x"], b["y"] - o["y"]) for o in allv if o["track_id"] != b["track_id"]]
            nn[(f, b["track_id"])] = min(d) if d else np.inf
    out = []
    for meth, fn in FILES.items():
        p = f"{WORK}/{key}/{tag}/{fn}"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); idc = "majority_object_id" if "majority_object_id" in df.columns else "track_id"
        df = df[(df.label == 0) & df[idc].notna() & df.frame_id.isin(list(gt_all))]  # AB3DMOT coasts past the last frame
        P = {f: g[["x", "y", idc]].to_numpy() for f, g in df.groupby("frame_id")}
        last, state = {}, {}; tp = fn_ = fp = ids = frag = 0; pairs = Counter(); gtc = Counter(); pc = Counter(); events = []
        box_strata = Counter(); tp_strata = Counter()
        for f in sorted(set(G) | set(P)):
            g = G.get(f, []); pr = P.get(f, np.zeros((0, 3)))
            for b in g: gtc[b["track_id"]] += 1; box_strata[np.digitize(nn[(f, b["track_id"])], BINS) - 1] += 1
            for r in pr: pc[int(r[2])] += 1
            mt = {}
            if len(g) and len(pr):
                C = np.array([[np.hypot(b["x"] - r[0], b["y"] - r[1]) for r in pr] for b in g]); C[C > MATCH] = 1e6
                ri, ci = linear_sum_assignment(C)
                mt = {g[i]["track_id"]: int(pr[j][2]) for i, j in zip(ri, ci) if C[i, j] <= MATCH}
            tp += len(mt); fn_ += len(g) - len(mt); fp += len(pr) - len(mt)
            for b in g:
                k = b["track_id"]
                if k in mt:
                    pairs[(k, mt[k])] += 1; tp_strata[np.digitize(nn[(f, k)], BINS) - 1] += 1
                    if k in last and last[k] != mt[k]:
                        ids += 1
                        near_merge = any((ff, k) in merged for ff in range(f - 3, f + 1))
                        events.append(dict(nn=nn[(f, k)], merge=near_merge))
                    if state.get(k) == "lost": frag += 1
                    last[k] = mt[k]; state[k] = "on"
                elif state.get(k) == "on": state[k] = "lost"
        gk, pk = sorted(gtc), sorted(pc)
        M = np.zeros((len(gk), max(len(pk), 1))); gi = {k: i for i, k in enumerate(gk)}; pi = {k: i for i, k in enumerate(pk)}
        for (a, b), v in pairs.items(): M[gi[a], pi[b]] = v
        r, c = linear_sum_assignment(-M); idtp = M[r, c].sum()
        E = pd.DataFrame(events) if events else pd.DataFrame(columns=["nn", "merge"])
        row = dict(scenario=key, tag=tag, method=meth, GT=sum(gtc.values()), TP=tp, FN=fn_, FP=fp, IDS=ids, Frag=frag, IDTP=idtp,
                   Pred=sum(pc.values()), Tracks=len(gk), IDS_merge=int(E["merge"].sum()) if len(E) else 0)
        for s in range(4):
            row[f"box_{BLAB[s]}"] = box_strata[s]; row[f"tp_{BLAB[s]}"] = tp_strata[s]
            row[f"ids_{BLAB[s]}"] = int(((np.digitize(E["nn"].astype(float), BINS) - 1) == s).sum()) if len(E) else 0
        out.append(row)
    # tracker-independent merge statistics
    vb = [(f, b["track_id"]) for f, v in G.items() for b in v]
    ms = dict(scenario=key, tag=tag, method="_front_end", GT=len(vb), merged=sum(1 for x in vb if x in merged))
    for s in range(4):
        sel = [x for x in vb if np.digitize(nn[x], BINS) - 1 == s]
        ms[f"box_{BLAB[s]}"] = len(sel); ms[f"merged_{BLAB[s]}"] = sum(1 for x in sel if x in merged)
    out.append(ms)
    return out

if __name__ == "__main__":
    keys = sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/*__*"))
    jobs = [(k, t) for k in keys for t in os.environ.get("TAGS", "eps0.5,eps0.8").split(",") if os.path.exists(f"{WORK}/{k}/{t}/members.pkl")]
    with Pool(28) as p: rows = [r for rr in p.map(one, jobs) for r in rr]
    pd.DataFrame(rows).to_csv(f"{RESULTS}/vru_eval_raw_m{MATCH}{os.environ.get('OUT', '')}.csv", index=False); print("saved", len(rows))
