"""Self-training baseline: CPU-side detector inputs of the evaluation clips (as lumpi_det.py, no foreground), so the
self-trained model can be inferred in the cloud. Run once with the M4 environment (LWORK=work_lumpi) and once with the
M6 environment (LWORK=work_lumpi_m6). Writes st_pkg/st_eval_in/<measurement>__<clip>/<frame index>.npz."""
import os, sys
from multiprocessing import Pool
import numpy as np
from lumpi_prep import ply, xyz, sensors, HERE, WORK
from lumpi_det import sensor_ground
TAG = sys.argv[1]   # m4 | m6

def one(job):
    out, i, src = job; p = ply(int(src)); X = xyz(p); d = {}
    for s, o in S.items():
        m = p["id"] == s
        if not m.any(): continue
        off = np.array([o[0], o[1], G[s]]); q = X[m] - off
        k = (np.abs(q[:, 0]) < 53.5) & (np.abs(q[:, 1]) < 53.5) & (q[:, 2] > -2.9) & (q[:, 2] < 4.5); m = np.flatnonzero(m)[k]; q = q[k]
        d[f"arr_{s}"] = np.column_stack([q, np.linalg.norm(X[m] - o, axis=1), p["intensity"][m].astype(np.float64)]).astype(np.float32); d[f"off_{s}"] = off
    np.savez_compressed(f"{out}/{i:04d}.npz", **d); return i

S = sensors(); G = sensor_ground()
if __name__ == "__main__":
    jobs = []
    for c in sorted(x for x in os.listdir(WORK) if x.startswith("lumpi__")):
        out = f"{HERE}/st_pkg/st_eval_in/{TAG}__{c}"; os.makedirs(out, exist_ok=True)
        jobs += [(out, i, s) for i, s in enumerate(np.load(f"{WORK}/{c}/frames.npy"))]
    with Pool(12) as pool: r = pool.map(one, jobs)
    print(TAG, len(r), "frames", "ground", {k: round(v, 3) for k, v in G.items()})
