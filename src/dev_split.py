"""Over-splitting of isolated VRUs by a fusion output tag (dev criterion for the DEDUP count fix, 2026-11-11).
Per labeled VRU (>= 3 returns) per frame: split = its returns fall in >= 2 output parts with >= 3 of its returns each
(members.pkl owners). Isolated = nearest other labeled VRU >= 1.5 m. Usage: WORK=... TAGS=a,b SEL=regex python dev_split.py"""
import glob, os, pickle, re
import numpy as np, pandas as pd
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru")
SEL = re.compile(os.environ.get("SEL", "^train[1-4]__")); TAGS = os.environ["TAGS"].split(",")
rows = []
for d in sorted(glob.glob(f"{WORK}/*__*")):
    if not SEL.search(os.path.basename(d)): continue
    G = pickle.load(open(f"{d}/gt_all.pkl", "rb"))
    nn = {}
    for f, bs in G.items():
        v = [b for b in bs if b["category_id"] == 0]
        for b in v:
            if b["npts"] >= 3: nn[(f, b["track_id"])] = min([np.hypot(b["x"] - o["x"], b["y"] - o["y"]) for o in v if o is not b] or [np.inf])
    for t in TAGS:
        if not os.path.exists(f"{d}/{t}/members.pkl"): continue
        parts = defaultdict(int)
        for m in pickle.load(open(f"{d}/{t}/members.pkl", "rb")):
            for k, n in m["owners"].items():
                if n >= 3: parts[(m["frame_id"], k)] += 1
        for key, dist in nn.items():
            rows.append(dict(tag=t, iso=dist >= 1.5, split=parts.get(key, 0) >= 2))
R = pd.DataFrame(rows)
print(R.groupby(["tag", "iso"]).split.mean().mul(100).round(2).unstack().rename(columns={True: "isolated_split_%", False: "near_split_%"}))
