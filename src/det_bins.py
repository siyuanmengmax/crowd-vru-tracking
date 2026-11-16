"""Per-point "clusters" for the detector-only tags, so MulTrack3D (which votes over the points of each detection) can run
on them: each box takes the real-foreground returns inside it (real0.5/detection, same in_box test as make_detonly.py);
a return inside several boxes goes to the nearest box centre. Boxes without returns get no points (MulTrack3D cannot use
them). Writes <tag>/detection/<frame>.bin in the cluster format (9 point columns + cluster id = row's cluster_id).
Usage: [WORK=...] TAGS=det0.45,ftvxn0.45 python det_bins.py"""
import glob, os
from multiprocessing import Pool
import numpy as np, pandas as pd
from make_detonly import in_box, WORK
TAGS = os.environ.get("TAGS", "det0.45,ftvxn0.45").split(",")

def one(job):
    key, tag = job; d = f"{WORK}/{key}/{tag}"; det = pd.read_csv(f"{d}/detection_bbox_results.csv"); empty = n = 0
    for f in sorted(glob.glob(f"{WORK}/{key}/real0.5/detection/*.bin")):
        name = os.path.basename(f); fid = int(name[:6]); pts = np.fromfile(f, np.float32).reshape(-1, 10)[:, :9].astype(np.float64)
        g = det[det.frame_id == fid]; best = np.full(len(pts), np.inf); cid = np.full(len(pts), -1)
        for b in g.itertuples():
            m = in_box(pts, b); dd = np.hypot(pts[:, 0] - b.x, pts[:, 1] - b.y); t = m & (dd < best); best[t] = dd[t]; cid[t] = b.cluster_id
        k = cid >= 0; n += len(g); empty += len(g) - len(np.unique(cid[k]))
        np.column_stack([pts[k], cid[k]]).astype(np.float32).tofile(f"{d}/detection/{name}")
    return key, tag, n, empty

if __name__ == "__main__":
    jobs = [(os.path.basename(p), t) for p in sorted(glob.glob(f"{WORK}/*__*")) for t in TAGS if os.path.isdir(f"{p}/{t}") and os.path.isdir(f"{p}/real0.5")]
    with Pool(24) as p: res = p.map(one, jobs)
    for t in TAGS:
        r = [x for x in res if x[1] == t]; print(t, len(r), "scenarios; boxes", sum(x[2] for x in r), "without returns", sum(x[3] for x in r))
