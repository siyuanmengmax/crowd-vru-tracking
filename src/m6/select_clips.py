"""M6 held-out clip selection (rule agreed with the user 2026-11-12, BEFORE any M6 point cloud or method output was read).
Labels only. 30-s windows (300 frames at 10 Hz) starting at multiples of 300; the trailing partial window is dropped.
Rank windows by the number of VRU box-frames (classes 0 person, 2 bicycle, 3 motorbike) whose nearest other VRU is < 0.8 m.
Crowded = top 3; medium = the 2 windows at the middle ranks (n//2 - 1, n//2 in descending order); sparse = the window with
the fewest such boxes among windows with >= 5 VRUs per frame (avoids empty windows). Output clips_m6.csv."""
import os
import pandas as pd, numpy as np
from scipy.spatial import cKDTree
d = pd.read_csv(os.environ["M6_LABEL"], header=None, skiprows=1, usecols=[0, 1, 7, 9, 10])
d.columns = ["t", "id", "cls", "x", "y"]; v = d[d.cls.isin([0, 2, 3])].copy(); v["f"] = (v.t * 10).round().astype(int)
nn = np.full(len(v), np.inf); xy = v[["x", "y"]].to_numpy()
for f, ii in v.groupby("f").indices.items():
    if len(ii) > 1: dd, _ = cKDTree(xy[ii]).query(xy[ii], k=2); nn[ii] = dd[:, 1]
v["close"] = nn < 0.8; v["win"] = v.f // 300
nfull = int((d.t.max() * 10 + 1) // 300)
g = v[v.win < nfull].groupby("win").agg(close=("close", "sum"), vru_per_frame=("f", lambda s: len(s) / 300), ids=("id", "nunique"))
g = g.reindex(range(nfull), fill_value=0)
r = g.sort_values(["close", "vru_per_frame"], ascending=False); n = len(r)
crowded = list(r.index[:3]); medium = list(r.index[[n // 2 - 1, n // 2]])
sp = g[g.vru_per_frame >= 5].sort_values(["close", "vru_per_frame"]); sparse = [sp.index[0]]
rows = [dict(clip=f"m6__c{w*300:04d}", start_frame=w * 300, stratum=s, **g.loc[w].round(1).to_dict())
        for s, ws in (("crowded", crowded), ("medium", medium), ("sparse", sparse)) for w in ws]
out = pd.DataFrame(rows); out.to_csv("clips_m6.csv", index=False); print(f"{nfull} full windows"); print(out.to_string(index=False))
