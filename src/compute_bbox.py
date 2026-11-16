"""Box fitting used by the FRGB3D pipeline (Meng et al., J. Comput. Civ. Eng. 2027), unchanged."""
import numpy as np
from numba import jit, prange

# @jit(nopython=True, cache=True)
def compute_bbox(points_coords):
    """Box with the smallest x-y footprint that strictly encloses all points."""
    if len(points_coords) == 0:
        # empty point set
        return [0, 0, 0, 0, 0, 0, 0]
    min_coords = np.min(points_coords, axis=0)  # minimum coordinates of the points
    max_coords = np.max(points_coords, axis=0)  # maximum coordinates of the points
    centroid_coords = (min_coords + max_coords) / 2  # centre of the points
    # centroid_coords = np.mean(points_coords, axis=0)  # centre of the points
    centered_points_coords = points_coords - centroid_coords  # move the points to the centre
    centered_points_2d_coords = centered_points_coords[:, :2]
    best_yaw = 0.0
    min_area = float('inf')
    for angle_deg in np.arange(0, 90, 0.1):  # all candidate rotation angles
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        rotated_points_coords = np.dot(centered_points_2d_coords, R)
        min_x, min_y = np.min(rotated_points_coords, axis=0)
        max_x, max_y = np.max(rotated_points_coords, axis=0)
        width = max_x - min_x
        length = max_y - min_y
        area = width * length
        if area < min_area:
            min_area = area
            best_yaw = angle
    # full 3D box at the best angle
    c, s = np.cos(best_yaw), np.sin(best_yaw)
    R_3d = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    # rotate all points by the best angle
    rotated_points_3d_coords = np.dot(centered_points_coords, R_3d)
    # axis-aligned box after rotation
    rot_min_coords = np.min(rotated_points_3d_coords, axis=0)
    rot_max_coords = np.max(rotated_points_3d_coords, axis=0)
    # exact box size (no extra margin)
    size = rot_max_coords - rot_min_coords
    # box centre (in the rotated frame)
    bbox_center_rotated_coords = (rot_min_coords + rot_max_coords) / 2
    # box centre back in the original frame
    bbox_center_coords = centroid_coords + np.dot(bbox_center_rotated_coords, R_3d.T)
    # check that the box contains all points (with a very small tolerance)
    tolerance = 1e-6  # tolerance for floating-point precision
    # all points in the box frame
    points_bbox_coords = np.dot(points_coords - bbox_center_coords, R_3d)
    # points inside the box?
    half_size = size / 2
    points_outside = np.any(np.abs(points_bbox_coords) > half_size + tolerance, axis=1)
    # if some points are outside, enlarge the box exactly
    if np.any(points_outside):
        # required enlargement per dimension
        min_vals = np.min(points_bbox_coords, axis=0)
        max_vals = np.max(points_bbox_coords, axis=0)
        # adjustment per dimension
        size_adjustment = np.zeros(3)
        for i in range(3):
            if min_vals[i] < -half_size[i] - tolerance:
                size_adjustment[i] += 2 * ((-half_size[i] - tolerance) - min_vals[i])
            if max_vals[i] > half_size[i] + tolerance:
                size_adjustment[i] += 2 * (max_vals[i] - (half_size[i] + tolerance))
        # update the size
        size += size_adjustment
    # final box parameters [x,y,z,w,l,h,yaw]
    bbox = [
        bbox_center_coords[0],
        bbox_center_coords[1],
        bbox_center_coords[2],
        size[0],  # width
        size[1],  # length
        size[2],  # height
        best_yaw
    ]
    return bbox
