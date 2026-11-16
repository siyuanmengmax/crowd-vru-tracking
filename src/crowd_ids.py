# Crowd-stratified IDS (raw-label eval_vru; IDS and per-stratum box counts do not depend on ignore regions).
import os, sys; sys.path.insert(0, os.getcwd())
import pandas as pd
from multiprocessing import Pool
import eval_vru as E
L = "lumpi" in E.WORK
tags = os.environ.get("TAGS", ",".join(["hyb_v1t45", "hyb_v1ftt45", "hyb_v1", "det0.45f", "ftvxn0.45f", "ftvxn0.45", "real0.5s"])).split(",")
keys = sorted(k for k in os.listdir(E.WORK) if "__" in k and (L or k.split("__")[0] in ("val", "test")))
jobs = [(k, t) for k in keys for t in tags if os.path.exists(f"{E.WORK}/{k}/{t}/members.pkl")]
with Pool(24) as p: d = pd.DataFrame([r for rr in p.map(E.one, jobs) for r in rr])
name = "lumpi" if L else "v2x"; d.to_csv(f"{E.RESULTS}/vru_eval_crowd_{name}{os.environ.get('OUT', '')}.csv", index=False)
d = d[d.method != "_front_end"]; d = d[d.method.isin(["AB3DMOT", "SinglePoint", "MulTrack3D", "MT_framelabel"])]; t = d.groupby(["method", "tag"]).sum(numeric_only=True)
t["Rec"] = 100 * t.TP / t.GT; t["IDS/kTP"] = 1000 * t.IDS / t.TP
for b in E.BLAB: t[f"R {b}"] = 100 * t[f"tp_{b}"] / t[f"box_{b}"]; t[f"IDS/kTP {b}"] = 1000 * t[f"ids_{b}"] / t[f"tp_{b}"]
pd.set_option("display.width", 250); print(name, "boxes per stratum", {b: int(t[f'box_{b}'].iloc[0]) for b in E.BLAB})
print(t[["GT", "Rec", "IDS", "IDS/kTP"] + [f"R {b}" for b in E.BLAB] + [f"IDS/kTP {b}" for b in E.BLAB]].round(2).to_string())
