"""Association-only and per-trajectory identity measures , on existing tracker outputs.
Same matching as eval_vru.py (BEV centre <= 1.0 m, Hungarian per frame, GT = VRU boxes with >= 3 foreground returns,
predictions = tracker boxes labelled VRU). Neither measure uses false positives, so both are identical with ignore regions.
 AssA_m  HOTA-style association accuracy over matched detections only: for each true positive c = (gt k, pred b),
         A(c) = TPA / (TPA + FNA + FPA), TPA = frames k matched to b, FNA = frames k matched to other predictions,
         FPA = frames b matched to other GT; AssA_m = mean of A over all true positives (detection recall excluded).
 Cover80 share of GT trajectories (>= 10 frames with >= 3 returns) of which one predicted ID covers >= 80% of those frames.
Ablation "no splitting" = hyb_ad8_nosplit_a20h3 (as in the paper; hyb_knosplit is identical to hyb_kde, SPLIT=0 has no effect
on the KDE count). Writes vru_assoc.csv (per sequence counts and sums)."""
import os, glob, pickle
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__)); MATCH = 1.0
RESULTS = os.environ.get("RESULTS", os.path.join(HERE, "..", "results")); os.makedirs(RESULTS, exist_ok=True)
FILES = {"AB3DMOT": "AB3DMOT_matched.csv", "Immortal": "Immortal_matched.csv", "SinglePoint": "SinglePointUKF_matched.csv"}
SITES = {"V2X-Real": ("work_vru", ("val__", "test__")), "M4": ("work_lumpi", ("lumpi__",)), "M6": ("work_lumpi_m6", ("lumpi__",))}
TAGS = {"AB3DMOT": ["hyb_kde_a20h3", "det0.45f_a20h3", "ftvxn0.45f_a4h3", "ftvxn0.45f_a20h3", "stvxn0.45f_a20h3", "real0.5s_a10h3",
                    "hyb_knomem_a20h3", "hyb_ad8_nosplit_a20h3", "hyb_v1d8_a20h3"],
        "Immortal": ["hyb_kde_im3", "det0.45f_im3", "ftvxn0.45f_im3", "stvxn0.45f_im3", "real0.5s_im3"],
        "SinglePoint": ["hyb_kde", "ftvxn0.45f"]}

def one(job):
    site, work, key, tag, meth = job; p = f"{work}/{key}/{tag}/{FILES[meth]}"
    G = pickle.load(open(f"{work}/{key}/gt_all.pkl", "rb"))
    df = pd.read_csv(p); idc = "majority_object_id" if "majority_object_id" in df.columns else "track_id"
    df = df[(df.label == 0) & df[idc].notna() & df.frame_id.isin(list(G))]
    P = {f: g[["x", "y", idc]].to_numpy() for f, g in df.groupby("frame_id")}
    pairs, gtc, tpk, tpb = Counter(), Counter(), Counter(), Counter(); last = {}; ids = 0
    for f in sorted(G):
        bs = G[f]
        g = [b for b in bs if b["category_id"] == 0 and b["npts"] >= 3]; pr = P.get(f)
        for b in g: gtc[b["track_id"]] += 1
        if not g or pr is None or not len(pr): continue
        C = np.hypot(np.array([b["x"] for b in g])[:, None] - pr[:, 0], np.array([b["y"] for b in g])[:, None] - pr[:, 1]); C[C > MATCH] = 1e6
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] <= MATCH: k, b = g[i]["track_id"], int(pr[j, 2]); pairs[(k, b)] += 1; tpk[k] += 1; tpb[b] += 1; ids += k in last and last[k] != b; last[k] = b
    asum = sum(n * n / (tpk[k] + tpb[b] - n) for (k, b), n in pairs.items())
    dom = Counter()
    for (k, b), n in pairs.items(): dom[k] = max(dom[k], n)
    long = [k for k, n in gtc.items() if n >= 10]
    return dict(site=site, scenario=key, tag=tag, method=meth, GT=sum(gtc.values()), TP=sum(tpk.values()), IDS=ids, Asum=asum,
                Traj=len(long), Cover80=sum(dom[k] >= 0.8 * gtc[k] for k in long))

if __name__ == "__main__":
    jobs = []
    for site, (w, pre) in SITES.items():
        W = os.environ.get({"V2X-Real": "WORK_V2X", "M4": "WORK_M4", "M6": "WORK_M6"}[site], f"{HERE}/{w}")
        for key in sorted(k for k in os.listdir(W) if k.startswith(pre)):
            for meth, tags in TAGS.items():
                for t in tags:
                    if os.path.exists(f"{W}/{key}/{t}/{FILES[meth]}"): jobs.append((site, W, key, t, meth))
    with Pool(16) as p: rows = p.map(one, jobs)
    d = pd.DataFrame(rows); d.to_csv(f"{RESULTS}/vru_assoc.csv", index=False)
    g = d.groupby(["site", "method", "tag"]).sum(numeric_only=True)
    g["AssA_m"] = 100 * g.Asum / g.TP; g["Cover80"] = 100 * g.Cover80 / g.Traj; g["Rec"] = 100 * g.TP / g.GT
    g["IDSk"] = 1000 * g.IDS / g.TP
    print(g[["AssA_m", "Cover80", "Traj", "Rec", "IDSk"]].round(1).to_string())
