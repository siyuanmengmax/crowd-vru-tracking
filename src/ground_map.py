"""Label-free local ground height for the V2X-Real intersection (the site slopes by about 5 m).
Stable background voxels (occupancy >= 0.01, bg_build.py; ground is sparse at range) of both LiDARs are moved to the world frame; the ground of a
CELL x CELL m cell is the lowest stable voxel, cleaned by a 3x3 median and filled from neighbours where empty.
Saves ground.npz (x0, y0, cell, z grid); ground_z(x, y) looks heights up."""
import glob, os, sys
import numpy as np, yaml
from scipy.ndimage import median_filter, distance_transform_edt
from v2x_prep import pose_matrix, SRC
HERE = os.path.dirname(os.path.abspath(__file__)); VOX = 0.2; CELL = 1.0; X0, X1, Y0, Y1 = -20, 110, -235, -95

def decode(k):
    m = (1 << 17) - 1
    return (np.c_[(k >> 34) & m, (k >> 17) & m, k & m].astype(np.int64) - 2 ** 15 + 0.5) * VOX

def build():
    nx, ny = int((X1 - X0) / CELL), int((Y1 - Y0) / CELL); G = np.full((nx, ny), np.inf)
    for ag in ("-1", "-2"):
        d = np.load(f"{HERE}/bg_{ag}.npz"); v = decode(d["keys"][d["occ"] >= 0.01])
        y = yaml.load(open(sorted(glob.glob(f"{SRC}/*/*/{ag}/*.yaml"))[0]), Loader=yaml.CSafeLoader); M = pose_matrix(y["lidar_pose"])
        w = v @ M[:3, :3].T + M[:3, 3]; i = ((w[:, 0] - X0) / CELL).astype(int); j = ((w[:, 1] - Y0) / CELL).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny); np.minimum.at(G, (i[ok], j[ok]), w[ok, 2] - VOX / 2)
    empty = ~np.isfinite(G); G[empty] = np.nanmedian(G[~empty])
    G = median_filter(G, 3); G[empty] = np.inf
    idx = distance_transform_edt(empty, return_distances=False, return_indices=True); G = G[tuple(idx)]
    np.savez(f"{HERE}/ground.npz", G=G.astype(np.float32), x0=X0, y0=Y0, cell=CELL); return G

_G = None
def ground_z(x, y):
    global _G
    if _G is None: _G = np.load(f"{HERE}/ground.npz")["G"]
    i = np.clip(((np.asarray(x) - X0) / CELL).astype(int), 0, _G.shape[0] - 1); j = np.clip(((np.asarray(y) - Y0) / CELL).astype(int), 0, _G.shape[1] - 1)
    return _G[i, j]

if __name__ == "__main__":
    import pickle
    build(); B = []
    for p in sorted(glob.glob(f"{HERE}/work_vru/*__*"))[::4]:
        gt = pickle.load(open(p + "/gt_all.pkl", "rb")); B += [(b["x"], b["y"], b["z"] - b["h"] / 2) for v in gt.values() for b in v if b["category_id"] in (0, 1)]
    B = np.array(B); e = B[:, 2] - ground_z(B[:, 0], B[:, 1])
    print("annotated box bottom - ground map: quantiles 5/25/50/75/95 =", np.percentile(e, [5, 25, 50, 75, 95]).round(2), "n", len(B))
