"""VRU tracking evaluation with ignore regions (KITTI DontCare style) for people the V2X-Real labels miss.
Same matching as eval_vru.py (BEV centre <= MATCH m, Hungarian, GT = VRU boxes with >= 3 returns). Ignore regions come
from the reviewed candidate tracks (track_candidates.json + tracks_review/results/joined.csv): a candidate judged
"person" or "annotated" (= a person in part of the track) marks its position in each of its frames. A prediction left
unmatched by the GT matching that lies within MATCH of such a position is dropped (neither TP nor FP, and not counted
in Pred for IDF1). Candidates judged "not" stay FP. Unreviewed candidates (short tracks outside the random sample, and
any skipped item): their unmatched predictions are counted separately as FP_unj (round 1) / FP_unj2 (round 2).
  conservative: FP = FP_rest + FP_unj
  corrected:    FP = FP_rest + FP_unj * (1 - p_short), p_short = share of people among the reviewed random short sample
Writes vru_eval_ign_m<MATCH>.csv (per scenario/tag/method: GT, TP, FN, FP_rest, FP_unj, IGN, IDS, Frag, IDTP, Pred_rest, Pred_unj)."""
import json, os, glob, pickle
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from eval_vru import FILES, WORK, HERE, MATCH, ANNOT, RESULTS
# review pools: round 1 (all earlier methods) and round 2 (candidates only the fine-tuned detectors produced)
POOLS = [(pd.read_csv(f"{ANNOT}/{d}/results/joined.csv"), json.load(open(f"{ANNOT}/{c}"))) for d, c in
         (("tracks_review", "track_candidates.json"), ("tracks_review2", "track_candidates_ft.json"))]
P_SHORT = [rev[rev.group == "short_sample"].answer.isin(["person", "annotated"]).mean() for rev, _ in POOLS]
REV = pd.concat([p[0] for p in POOLS]); CANDS = POOLS[0][1]

def regions(key, pool):
    rev, cands = POOLS[pool]; ans = {int(r.cand): r.answer for r in rev[rev.scenario == key].itertuples()}
    ign, unj = {}, {}
    for ci, o in enumerate(cands.get(key, [])):
        a = ans.get(ci); a = None if pd.isna(a) else a   # unreviewed or skipped
        tgt = ign if a in ("person", "annotated") else unj if a is None else None
        if tgt is None: continue
        for f, v in o["path"].items(): tgt.setdefault(int(f), []).append(v[:2])
    return {f: np.array(v) for f, v in ign.items()}, {f: np.array(v) for f, v in unj.items()}

def near(r, pts):
    return pts is not None and len(pts) and np.hypot(pts[:, 0] - r[0], pts[:, 1] - r[1]).min() <= MATCH

def one(job):
    key, tag = job
    gt_all = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb")); (I1, U1), (I2, U2) = regions(key, 0), regions(key, 1)
    IGN = {f: np.vstack([d[f] for d in (I1, I2) if f in d]) for f in set(I1) | set(I2)}
    G = {f: [b for b in bs if b["category_id"] == 0 and b["npts"] >= 3] for f, bs in gt_all.items()}
    out = []
    for meth, fn in FILES.items():
        p = f"{WORK}/{key}/{tag}/{fn}"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); idc = "majority_object_id" if "majority_object_id" in df.columns else "track_id"
        df = df[(df.label == 0) & df[idc].notna() & df.frame_id.isin(list(gt_all))]
        P = {f: g[["x", "y", idc]].to_numpy() for f, g in df.groupby("frame_id")}
        last, state = {}, {}; c = Counter(); pairs = Counter(); gtc = Counter(); pc = Counter()
        for f in sorted(set(G) | set(P)):
            g = G.get(f, []); pr = P.get(f, np.zeros((0, 3)))
            for b in g: gtc[b["track_id"]] += 1
            mt, used = {}, set()
            if len(g) and len(pr):
                C = np.array([[np.hypot(b["x"] - r[0], b["y"] - r[1]) for r in pr] for b in g]); C[C > MATCH] = 1e6
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] <= MATCH: mt[g[i]["track_id"]] = int(pr[j][2]); used.add(j)
            for j, r in enumerate(pr):
                if j in used: pc[int(r[2])] += 1
                elif near(r, IGN.get(f)): c["IGN"] += 1
                elif near(r, U1.get(f)): c["FP_unj"] += 1; c["Pred_unj"] += 1
                elif near(r, U2.get(f)): c["FP_unj2"] += 1; c["Pred_unj2"] += 1
                else: c["FP_rest"] += 1; pc[int(r[2])] += 1
            c["TP"] += len(mt); c["FN"] += len(g) - len(mt)
            for b in g:
                k = b["track_id"]
                if k in mt:
                    pairs[(k, mt[k])] += 1
                    if k in last and last[k] != mt[k]: c["IDS"] += 1
                    if state.get(k) == "lost": c["Frag"] += 1
                    last[k] = mt[k]; state[k] = "on"
                elif state.get(k) == "on": state[k] = "lost"
        gk, pk = sorted(gtc), sorted(pc)
        M = np.zeros((len(gk), max(len(pk), 1))); gi = {k: i for i, k in enumerate(gk)}; pi = {k: i for i, k in enumerate(pk)}
        for (a, b), v in pairs.items(): M[gi[a], pi[b]] = v
        r, cc = linear_sum_assignment(-M)
        out.append(dict(scenario=key, tag=tag, method=meth, GT=sum(gtc.values()), IDTP=M[r, cc].sum(), Pred_rest=sum(pc.values()),
                        **{k: c[k] for k in ("TP", "FN", "FP_rest", "FP_unj", "Pred_unj", "FP_unj2", "Pred_unj2", "IGN", "IDS", "Frag")}))
    return out

def summary(df):
    t = df.groupby(["method", "tag"]).sum(numeric_only=True)
    for name, w1, w2 in (("cons", 1.0, 1.0), ("corr", 1 - P_SHORT[0], 1 - P_SHORT[1])):
        fp = t.FP_rest + w1 * t.FP_unj + w2 * t.FP_unj2; pred = t.Pred_rest + w1 * t.Pred_unj + w2 * t.Pred_unj2
        t[f"MOTA_{name}"] = 100 * (1 - (t.FN + fp + t.IDS) / t.GT); t[f"IDF1_{name}"] = 100 * 2 * t.IDTP / (t.GT + pred)
        t[f"Prec_{name}"] = 100 * t.TP / (t.TP + fp)
    t["Rec"] = 100 * t.TP / t.GT
    return t

if __name__ == "__main__":
    keys = sorted(CANDS); tags = os.environ.get("TAGS", "det0.3,det0.45,real0.5s,hyb_v1,hyb_v2rmc,ftvxn0.3,ftvxn0.45,ftcp0.3,ftcp0.45").split(",")
    jobs = [(k, t) for k in keys for t in tags if os.path.isdir(f"{WORK}/{k}/{t}")]
    with Pool(16) as p: df = pd.DataFrame([r for rr in p.map(one, jobs) for r in rr])
    df.to_csv(f"{RESULTS}/vru_eval_ign_m{MATCH}{os.environ.get('OUT', '')}.csv", index=False)
    print("p_short =", [round(p, 3) for p in P_SHORT], "; reviewed", REV.answer.notna().sum(), "of", sum(sum(map(len, c.values())) for _, c in POOLS), "candidates")
    t = summary(df); pd.set_option("display.width", 250)
    print(t[["GT", "FN", "FP_rest", "FP_unj", "FP_unj2", "IGN", "IDS", "Rec", "Prec_cons", "Prec_corr", "MOTA_cons", "MOTA_corr", "IDF1_cons", "IDF1_corr"]].round(1).to_string())
