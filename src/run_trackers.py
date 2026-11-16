"""Single Point UKF baseline tracker on a front-end output tag (the UKF rows of the paper).
For every <WORK>/<scenario>/<tag>/detection_bbox_results.csv, runs tracking_single_point.SinglePointTracker
(max_distance 3 m, max_age 3) and matches its tracks back to the boxes -> SinglePointUKF_matched.csv.
Same call as in the original experiments; the MulTrack3D and ClusterTrack3D branches are not part of this release."""
import os, sys, glob, contextlib, io
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.environ.get("WORK", f"{HERE}/work_vru")
sys.path.insert(0, HERE)
METHODS = os.environ.get("METHODS", "SinglePoint+UKF").split(",")

def run(job):
    key, tag, method = job
    if method != "SinglePoint+UKF": raise ValueError(f"method {method} is not part of this release")
    d = f"{WORK}/{key}/{tag}"; out = f"{d}/SinglePointUKF_matched.csv"
    if os.path.exists(out) and not os.environ.get("FORCE"): return out
    os.chdir(d)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import tracking_single_point as sp
        t = sp.SinglePointTracker(csv_path="detection_bbox_results.csv", max_distance=3, max_age=3)
        if t.run(): t.postprocess_labels()
        t.save_results(out)
    return out

if __name__ == "__main__":
    keys = sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/*__*"))
    if os.environ.get("SPLITS"): keys = [k for k in keys if k.split("__")[0] in os.environ["SPLITS"].split(",")]   # e.g. SPLITS=val,test
    tags = os.environ.get("TAGS", "hyb_kde").split(",")
    jobs = [(k, t, m) for k in keys for t in tags for m in METHODS if os.path.isdir(f"{WORK}/{k}/{t}")]
    with Pool(int(os.environ.get("NPROC", 8))) as p:
        for r in p.imap_unordered(run, jobs): pass
    print("done", len(jobs))
