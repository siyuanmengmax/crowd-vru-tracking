"""Self-training baseline (protocol in the paper, Self-Training Baseline), CPU part: per selected M6 frame (outside the held-out clips, 100-frame buffer,
every 9th frame), the exact per-LiDAR detector input of lumpi_det.py before prepare_cloud (x, y, z relative to the sensor
ground, range, reflectivity), the sensor offsets, and the foreground returns (M6 background model, all LiDARs) for the
pseudo-label filter. The GPU part (st_pkg/st_label.py) runs in the cloud because this workstation's GPU fails in inference.
Run with the M6 environment. Writes st_pkg/st_in/<frame>.npz."""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
from lumpi_prep import ply, xyz, sensors, background, occ, THETA, HERE
from lumpi_det import sensor_ground
OUT = f"{HERE}/st_pkg/st_in"; STEP, BUF = 9, 100

def frames():
    starts = pd.read_csv(f"{HERE}/m6/clips_m6.csv").start_frame.tolist(); n = len(os.listdir(f"{os.environ['LSRC']}/lidar"))
    bad = np.zeros(n, bool)
    for s in starts: bad[max(0, s - BUF):s + 300 + BUF] = True
    return [f for f in range(0, n, STEP) if not bad[f]]

def one(src):
    p = ply(src); X = xyz(p); d = {"fg": X[occ(BG, X) < THETA].astype(np.float32)}
    for s, o in S.items():
        m = p["id"] == s
        if not m.any(): continue
        off = np.array([o[0], o[1], G[s]]); q = X[m] - off
        k = (np.abs(q[:, 0]) < 53.5) & (np.abs(q[:, 1]) < 53.5) & (q[:, 2] > -2.9) & (q[:, 2] < 4.5); m = np.flatnonzero(m)[k]; q = q[k]
        d[f"arr_{s}"] = np.column_stack([q, np.linalg.norm(X[m] - o, axis=1), p["intensity"][m].astype(np.float64)]).astype(np.float32)
        d[f"off_{s}"] = off
    np.savez_compressed(f"{OUT}/{src:06d}.npz", **d); return src

S = sensors(); G = sensor_ground(); BG = background()
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); F = frames()
    with Pool(12) as pool: r = pool.map(one, F)
    print("frames", len(r), "sensors", sorted(S), "ground", {k: round(v, 3) for k, v in G.items()}, "ST_CPU_DONE")
