"""Foreground-filtered detector baseline (tags <tag>f): keep only the boxes that contain at least one real-foreground
return (the boxes det_bins.py gave points to), i.e. the same box set MulTrack3D sees on the detector tags. Isolates the
effect of background filtering from the tracker. Usage: [WORK=...] TAGS=det0.45,ftvxn0.45 python det_fgfilter.py"""
import glob, os, pickle, shutil
from multiprocessing import Pool
import numpy as np, pandas as pd
from make_detonly import WORK
TAGS = os.environ.get("TAGS", "det0.45,ftvxn0.45").split(",")

def one(job):
    key, tag = job; s = f"{WORK}/{key}/{tag}"; d = f"{s}f"; os.makedirs(d, exist_ok=True)
    det = pd.read_csv(f"{s}/detection_bbox_results.csv"); keep = set()
    for f in glob.glob(f"{s}/detection/*.bin"):
        fid = int(os.path.basename(f)[:6]); c = np.fromfile(f, np.float32).reshape(-1, 10)[:, 9]
        keep |= {(fid, int(x)) for x in np.unique(c)}
    m = [(a, b) in keep for a, b in zip(det.frame_id, det.cluster_id)]; det[m].to_csv(f"{d}/detection_bbox_results.csv", index=False)
    mem = [x for x in pickle.load(open(f"{s}/members.pkl", "rb")) if (x["frame_id"], x["cluster_id"]) in keep]
    pickle.dump(mem, open(f"{d}/members.pkl", "wb"))
    if os.path.exists(f"{d}/detection"): shutil.rmtree(f"{d}/detection")
    shutil.copytree(f"{s}/detection", f"{d}/detection")
    return tag, len(det), sum(m)

if __name__ == "__main__":
    keys = [os.path.basename(p) for p in sorted(glob.glob(f"{WORK}/*__*")) if os.path.basename(p).split("__")[0] in ("val", "test", "lumpi")]
    jobs = [(k, t) for k in keys for t in TAGS if glob.glob(f"{WORK}/{k}/{t}/detection/*.bin")]
    with Pool(16) as p: r = p.map(one, jobs)
    for t in TAGS: x = [v for v in r if v[0] == t]; print(t, len(x), "scenarios; kept", sum(v[2] for v in x), "of", sum(v[1] for v in x))
