"""Real background model for the V2X-Real infrastructure LiDARs (step 2b).
Both LiDARs keep the same pose in all 58 scenarios, so one background per LiDAR is learned from every STEP-th frame
of all scenarios: a 0.2 m voxel in the LiDAR's own frame is background when it is occupied in >= theta of the frames.
Saves bg_<agent>.npz with the sorted voxel keys and their occupancy fraction."""
import glob, os, sys
import numpy as np
from v2x_prep import SRC
HERE = os.path.dirname(os.path.abspath(__file__)); BG_DIR = os.environ.get("BG_DIR", HERE); VOX = 0.2; STEP = int(os.environ.get("STEP", 4))

def keys(p):
    v = np.floor(p[:, :3] / VOX).astype(np.int64) + 2 ** 15
    return (v[:, 0] << 34) | (v[:, 1] << 17) | v[:, 2]

if __name__ == "__main__":
    for ag in ("-1", "-2"):
        files = sorted(glob.glob(f"{SRC}/*/*/{ag}/*.bin"))[::STEP]; acc = []
        for f in files:
            p = np.fromfile(f, np.float32).reshape(-1, 4); p = p[np.abs(p[:, :3]).sum(1) > 0]
            acc.append(np.unique(keys(p)))
        k, c = np.unique(np.concatenate(acc), return_counts=True)
        np.savez(f"{BG_DIR}/bg_{ag}.npz", keys=k, occ=(c / len(files)).astype(np.float32), n=len(files))
        print(ag, len(files), "frames,", len(k), "voxels; occupancy quantiles", np.quantile(c / len(files), [.5, .9, .99]).round(3))
