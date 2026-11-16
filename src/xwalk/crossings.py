"""Downstream demo (M6): pedestrian crossings at the LUMPI crosswalk on the north-east arm.
Geometry (from GT vehicle and VRU positions only, fixed before any method output was read): arm axis 47 deg through
the local origin; s along the arm, d across it. Lanes occupy d in about [-10, 7]; curb lines at D_LO, D_HI; crosswalk
window S_LO <= s <= S_HI. A crossing is a pass of one trajectory from one curb line to the other inside the window.
GT crossings: GT VRU tracks (category 0) with both curb passes inside the clip.
Per method (LUMPI, AB3DMOT, chosen track management):
  complete   one predicted ID is matched (Hungarian, 1 m, as the evaluation) to the person in >= COVER of the
             crossing frames in which the person has >= 3 returns
  dur_err    |predicted - GT crossing time| for GT crossings whose dominant predicted ID itself makes the crossing
  n_pred     crossing events in the predicted VRU tracks (includes unlabeled people, so read with care)."""
import glob, os, pickle, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
GLOB, SUF = os.environ.get("GLOB", "work_lumpi/lumpi__c*"), os.environ.get("SUF", "")
FN = os.environ.get("FN", "AB3DMOT_matched.csv")   # 2026-11-13: tracker output file (Immortal_matched.csv for ImmortalTracker)
TH, D_LO, D_HI, S_LO, S_HI, COVER, MATCH = np.deg2rad(47), -11.5, 8.5, 12.0, 26.0, 0.8, 1.0
TAGS = eval(os.environ["TAGS_X"]) if os.environ.get("TAGS_X") else {"Proposed": "hyb_v1t45_a10h3", "KDE count": "hyb_kde_a10h3", "No memory": "hyb_a_nomem_a10h3",
        "Zero-shot + filter": "det0.45f_a20h3", "Fine-tuned + filter": "ftvxn0.45f_a4h3", "Size rule": "real0.5s_a10h3"}

def sd(x, y): return np.cos(TH) * x + np.sin(TH) * y, -np.sin(TH) * x + np.cos(TH) * y

def events(fr, x, y):
    """Crossings of one trajectory: list of (f_start, f_end) between passes of opposite curb lines inside the window."""
    o = np.argsort(fr); fr, x, y = fr[o], x[o], y[o]; s, d = sd(x, y)
    side = np.where(d < D_LO, -1, np.where(d > D_HI, 1, 0)); out, last = [], None   # last = (side, frame) when it left a curb
    for i in range(len(fr)):
        if side[i] != 0:
            if last is not None and last[0] == -side[i] and S_LO <= s[i] <= S_HI and S_LO <= last[2] <= S_HI: out.append((last[1], fr[i]))
            last = (side[i], fr[i], s[i])
    return out

def run(tag):
    rows = []; npred = 0
    for d in sorted(glob.glob(GLOB)):
        G = pickle.load(open(f"{d}/gt_all.pkl", "rb"))
        gtr = defaultdict(list)
        for f, bs in G.items():
            for b in bs:
                if b["category_id"] == 0: gtr[b["track_id"]].append((f, b["x"], b["y"], b["npts"]))
        pr = pd.read_csv(f"{d}/{tag}/{FN}"); pr = pr[(pr.label == 0) & pr.frame_id.isin(list(G))]
        ptr = {k: g for k, g in pr.groupby("track_id")}
        pev = {k: events(g.frame_id.to_numpy(), g.x.to_numpy(), g.y.to_numpy()) for k, g in ptr.items()}
        npred += sum(len(v) for v in pev.values())
        PF = {f: g[["x", "y", "track_id"]].to_numpy() for f, g in pr.groupby("frame_id")}
        match = {}
        for f, bs in G.items():
            g = [b for b in bs if b["category_id"] == 0 and b["npts"] >= 3]; p = PF.get(f)
            if not g or p is None or not len(p): continue
            C = np.hypot(np.array([b["x"] for b in g])[:, None] - p[:, 0], np.array([b["y"] for b in g])[:, None] - p[:, 1]); C[C > MATCH] = 1e6
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] <= MATCH: match[(f, g[i]["track_id"])] = int(p[j, 2])
        for k, t in gtr.items():
            t = np.array(t)
            for f0, f1 in events(t[:, 0].astype(int), t[:, 1], t[:, 2]):
                fs = [int(f) for f, n in zip(t[:, 0], t[:, 3]) if f0 <= f <= f1 and n >= 3]
                ids = Counter(match.get((f, k)) for f in fs); ids.pop(None, None)
                top, n = ids.most_common(1)[0] if ids else (None, 0)
                comp = len(fs) > 0 and n >= COVER * len(fs)
                pe = [e for e in pev.get(top, []) if e[0] <= f1 and e[1] >= f0] if top is not None else []
                err = abs((pe[0][1] - pe[0][0]) - (f1 - f0)) / 10 if pe else np.nan
                rows.append(dict(clip=d[-5:], gt=k, f0=f0, f1=f1, dur=(f1 - f0) / 10, nfr=len(fs), nid=len(ids), complete=comp, dur_err=err))
    return pd.DataFrame(rows), npred

if __name__ == "__main__":
    out = []
    for name, tag in TAGS.items():
        r, npred = run(tag)
        out.append(dict(method=name, gt_crossings=len(r), complete_pct=round(100 * r.complete.mean(), 1),
                        ids_per_crossing=round(r.nid.mean(), 2), with_pred_crossing_pct=round(100 * r.dur_err.notna().mean(), 1),
                        dur_err_med_s=round(r.dur_err.median(), 2), n_pred_crossings=npred))
        r.to_csv(f"xwalk/cross{SUF}_{tag}.csv", index=False)
    o = pd.DataFrame(out); o.to_csv(f"xwalk/crossings_summary{SUF}.csv", index=False); print(o.to_string(index=False))
