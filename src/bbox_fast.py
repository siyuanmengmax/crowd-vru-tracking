"""compute_bbox (lowell_test/src/detection/compute_bbox.py) with the 0.1-degree yaw search vectorised: same grid, same first-minimum rule, same final box.
Only the convex-hull vertices enter the search, since the extremes along any direction lie on the hull."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError
_A = np.radians(np.arange(0, 90, 0.1)); _C, _S = np.cos(_A), np.sin(_A)

def compute_bbox(points_coords):
    if len(points_coords) == 0: return [0, 0, 0, 0, 0, 0, 0]
    lo, hi = points_coords.min(0), points_coords.max(0); ctr = (lo + hi) / 2; q = points_coords - ctr; p = q[:, :2]
    if len(p) > 16:
        try: p = p[ConvexHull(p).vertices]
        except QhullError: pass
    u = np.outer(_C, p[:, 0]) + np.outer(_S, p[:, 1]); v = np.outer(_C, p[:, 1]) - np.outer(_S, p[:, 0])
    best = _A[((u.max(1) - u.min(1)) * (v.max(1) - v.min(1))).argmin()]
    c, s = np.cos(best), np.sin(best); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    r = q @ R; a, b = r.min(0), r.max(0); size = b - a; centre = ctr + ((a + b) / 2) @ R.T
    return [centre[0], centre[1], centre[2], size[0], size[1], size[2], best]
