"""Paired sequence bootstrap (B 10,000, seed 20261108, as ci.py) for AssA_m, Cover80, IDS/kTP from vru_assoc.csv."""
import numpy as np, pandas as pd
import os
RESULTS = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
d = pd.read_csv(f"{RESULTS}/vru_assoc.csv").set_index(["site", "method", "tag", "scenario"]).sort_index()
B, SEED = 10000, 20261108
def met(c):  # columns GT TP IDS Asum Traj Cover80
    GT, TP, IDS, A, T, C = np.moveaxis(c, -1, 0)
    return {"AssA_m": 100 * A / TP, "Cover80": 100 * C / T, "IDS/kTP": 1000 * IDS / TP}
CMP = [("AB3DMOT", "hyb_kde_a20h3", b) for b in ("det0.45f_a20h3", "ftvxn0.45f_a4h3", "ftvxn0.45f_a20h3", "stvxn0.45f_a20h3", "hyb_knomem_a20h3", "hyb_ad8_nosplit_a20h3")] + \
      [("Immortal", "hyb_kde_im3", b) for b in ("det0.45f_im3", "ftvxn0.45f_im3", "stvxn0.45f_im3")] + [("SinglePoint", "hyb_kde", "ftvxn0.45f")]
rows = []
for site in ("V2X-Real", "M4", "M6"):
    for m, a, b in CMP:
        try: A, Bm = d.loc[(site, m, a)], d.loc[(site, m, b)]
        except KeyError: continue
        s = sorted(set(A.index) & set(Bm.index)); cols = ["GT", "TP", "IDS", "Asum", "Traj", "Cover80"]
        A, Bm = A.loc[s, cols].to_numpy(float), Bm.loc[s, cols].to_numpy(float); n = len(s)
        idx = np.random.default_rng(SEED).integers(0, n, (B, n)); pa, pb = met(A.sum(0)), met(Bm.sum(0))
        ba, bb = met(A[idx].sum(1)), met(Bm[idx].sum(1))
        for k in pa:
            lo, hi = np.percentile(ba[k] - bb[k], [2.5, 97.5])
            rows.append(dict(site=site, tracker=m, A=a, B=b, metric=k, vA=round(pa[k], 1), vB=round(pb[k], 1), delta=round(pa[k] - pb[k], 1),
                             ci_lo=round(lo, 1), ci_hi=round(hi, 1), excl0=bool(lo > 0 or hi < 0)))
o = pd.DataFrame(rows); o.to_csv(f"{RESULTS}/ci_results_assoc.csv", index=False)
pd.set_option("display.width", 200); print(o.to_string(index=False))
