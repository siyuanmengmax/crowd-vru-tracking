import glob, pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from crossings import sd, D_LO, D_HI, S_LO, S_HI, TH, TAGS
fig, a = plt.subplots(figsize=(9, 9))
import os; SUF = os.environ.get("SUF", ""); GLOB = os.environ.get("GLOB", "work_lumpi/lumpi__c*")
r = pd.read_csv(f"xwalk/cross{SUF}_hyb_v1t45_a10h3.csv")
for d in sorted(glob.glob(GLOB)):
    G = pickle.load(open(f"{d}/gt_all.pkl", "rb"))
    for row in r[r["clip"] == d[-5:]].itertuples():
        p = np.array([(b["x"], b["y"]) for f in range(row.f0 - 20, row.f1 + 20) for b in G.get(f, []) if b["track_id"] == row.gt])
        a.plot(p[:, 0], p[:, 1], lw=0.6, c="tab:blue" if row.complete else "tab:red")
    c = np.array([(b["x"], b["y"]) for f in list(G)[::5] for b in G[f] if b["category_id"] != 0]); a.scatter(c[:, 0], c[:, 1], s=0.2, c="gray")
for dd in (D_LO, D_HI):
    s = np.array([S_LO, S_HI]); a.plot(np.cos(TH) * s - np.sin(TH) * dd, np.sin(TH) * s + np.cos(TH) * dd, "k-", lw=2)
a.set_xlim(-5, 40); a.set_ylim(-5, 40); a.set_aspect("equal"); fig.savefig(f"xwalk/check{SUF}.png", dpi=70)
# paired bootstrap over crossings, clustered by clip
rng = np.random.default_rng(20261108)
base = r[["clip", "gt", "f0"]]
R = {n: pd.read_csv(f"xwalk/cross{SUF}_{t}.csv").merge(base, on=["clip", "gt", "f0"]).complete.to_numpy(float) for n, t in TAGS.items()}
clips = r["clip"].to_numpy(); uc = np.unique(clips); print("crossings per clip", {c: int((clips == c).sum()) for c in uc})
for n in R:
    if n == "Proposed": continue
    dlt = R["Proposed"] - R[n]; bs = []
    for _ in range(10000):
        pick = rng.choice(uc, len(uc)); idx = np.concatenate([np.where(clips == c)[0] for c in pick]); bs.append(100 * dlt[idx].mean())
    print(f"Proposed - {n}: {100*dlt.mean():+.1f} pp, CI {np.percentile(bs,2.5):+.1f} to {np.percentile(bs,97.5):+.1f}; P better {int((dlt>0).sum())}, worse {int((dlt<0).sum())}")
