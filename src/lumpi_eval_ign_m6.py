"""M6 copy (held-out recording, protocol fixed in advance as described in the paper): review and candidates from lumpi_review_m6.py, families by front end as there,
ImmortalTracker and MulTrack3D per-detection-label outputs included. Run with WORK=work_lumpi_m6. Otherwise identical to M4.

LUMPI tracking evaluation with ignore regions (eval_ign.py adapted to the stratified LUMPI review, lumpi_review.py).
Candidates judged "person"/"annotated" mark ignore regions (an unmatched prediction within MATCH is dropped); "not" stays
FP. Unreviewed candidates (outside the sample, or without a camera view) are counted per stratum g in FP_un_<g>:
  conservative: FP = FP_rest + sum_g FP_un_g
  corrected:    FP = FP_rest + sum_g FP_un_g * (1 - p_g), p_g = share of people among the judged items of stratum g
Run with WORK=work_lumpi.  Writes vru_eval_ign_lumpi.csv."""
import json, os, pickle
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from eval_vru import FILES, WORK, HERE, MATCH, ANNOT, RESULTS
G = ("L3", "L2", "L1", "S"); LONG = 30
REV = pd.read_csv(f"{ANNOT}/lumpi_m6_review/results/joined.csv"); CANDS = json.load(open(f"{ANNOT}/lumpi_m6_track_candidates.json"))
P = {g: REV[(REV.group == g) & REV.answer.notna()].answer.isin(["person", "annotated"]).mean() for g in G}

def family(src): return len({"clu" if t.startswith(("hyb_", "real0.5s")) else "zs" if t.startswith("det") else "ft" for t in src})

def regions(key):
    ans = {int(r.cand): r.answer for r in REV[REV.scenario == key].itertuples() if isinstance(r.answer, str)}
    ign, unj = {}, {g: {} for g in G}
    for ci, o in enumerate(CANDS[key]):
        a = ans.get(ci)
        if a == "not": continue
        g = "S" if len(o["path"]) < LONG else f"L{family(o['src'])}"
        tgt = ign if a in ("person", "annotated") else unj[g]
        for f, v in o["path"].items(): tgt.setdefault(int(f), []).append(v[:2])
    return {f: np.array(v) for f, v in ign.items()}, {g: {f: np.array(v) for f, v in d.items()} for g, d in unj.items()}

def near(r, pts):
    return pts is not None and len(pts) and np.hypot(pts[:, 0] - r[0], pts[:, 1] - r[1]).min() <= MATCH

def one(job):
    key, tag = job
    gt_all = pickle.load(open(f"{WORK}/{key}/gt_all.pkl", "rb")); IGN, U = regions(key)
    GT = {f: [b for b in bs if b["category_id"] == 0 and b["npts"] >= 3] for f, bs in gt_all.items()}
    out = []
    for meth, fn in FILES.items():
        p = f"{WORK}/{key}/{tag}/{fn}"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); idc = "majority_object_id" if "majority_object_id" in df.columns else "track_id"
        df = df[(df.label == 0) & df[idc].notna() & df.frame_id.isin(list(gt_all))]
        PR = {f: g[["x", "y", idc]].to_numpy() for f, g in df.groupby("frame_id")}
        last, state = {}, {}; c = Counter(); pairs = Counter(); gtc = Counter(); pc = Counter()
        for f in sorted(set(GT) | set(PR)):
            g = GT.get(f, []); pr = PR.get(f, np.zeros((0, 3)))
            for b in g: gtc[b["track_id"]] += 1
            mt, used = {}, set()
            if len(g) and len(pr):
                C = np.array([[np.hypot(b["x"] - r[0], b["y"] - r[1]) for r in pr] for b in g]); C[C > MATCH] = 1e6
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] <= MATCH: mt[g[i]["track_id"]] = int(pr[j][2]); used.add(j)
            for j, r in enumerate(pr):
                if j in used: pc[int(r[2])] += 1; continue
                if near(r, IGN.get(f)): c["IGN"] += 1; continue
                for s in G:
                    if near(r, U[s].get(f)): c[f"FP_un_{s}"] += 1; break
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
                        **{k: c[k] for k in ["TP", "FN", "FP_rest", "IGN", "IDS", "Frag"] + [f"FP_un_{s}" for s in G]}))
    return out

def summary(df):
    t = df.groupby(["method", "tag"]).sum(numeric_only=True)
    for name, w in (("cons", {s: 1.0 for s in G}), ("corr", {s: 1 - P[s] for s in G})):
        un = sum(w[s] * t[f"FP_un_{s}"] for s in G); fp = t.FP_rest + un; pred = t.Pred_rest + un
        t[f"MOTA_{name}"] = 100 * (1 - (t.FN + fp + t.IDS) / t.GT); t[f"IDF1_{name}"] = 100 * 2 * t.IDTP / (t.GT + pred)
        t[f"Prec_{name}"] = 100 * t.TP / (t.TP + fp)
    t["Rec"] = 100 * t.TP / t.GT; t["FP_un"] = sum(t[f"FP_un_{s}"] for s in G)
    return t

if __name__ == "__main__":
    tags = os.environ.get("TAGS", "real0.5s,hyb_v1,det0.3,det0.45,ftvxn0.3,ftvxn0.45").split(",")
    jobs = [(k, t) for k in sorted(CANDS) for t in tags if os.path.isdir(f"{WORK}/{k}/{t}")]
    with Pool(16) as p: df = pd.DataFrame([r for rr in p.map(one, jobs) for r in rr])
    df.to_csv(f"{RESULTS}/vru_eval_ign_lumpi_m6{os.environ.get('OUT', '')}.csv", index=False)
    print("p =", {s: round(v, 3) for s, v in P.items()}); t = summary(df); pd.set_option("display.width", 250)
    print(t[["GT", "FN", "FP_rest", "FP_un", "IGN", "IDS", "Rec", "Prec_cons", "Prec_corr", "MOTA_cons", "MOTA_corr", "IDF1_cons", "IDF1_corr"]].round(1).to_string())
