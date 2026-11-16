"""Late fusion of the boxes of two LiDARs and assignment of points to boxes

Steps:
1. each LiDAR is inferred in its own ground-levelled frame
2. boxes are transformed rigidly into the world frame (= the ground frame of the target LiDAR)
3. BEV IoU matching in the world frame: matched pairs are merged, single boxes are kept
4. points of the merged cloud get the cluster_id of their box, for the downstream tracker
"""

import numpy as np
from scipy.spatial import cKDTree


def transform_boxes(boxes, matrix):
    """Transform (K,7) boxes [x,y,z,dx,dy,dz,heading] into another frame with a 4x4 rigid matrix.

    The merge_matrix is measured to be an almost pure yaw rotation (tilt < 1 deg), so the heading
    is shifted by the yaw component of the matrix.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 7)
    out = boxes.copy()
    centers = np.column_stack([out[:, :3], np.ones(len(out))])
    out[:, :3] = (matrix @ centers.T).T[:, :3]
    out[:, 6] = out[:, 6] + np.arctan2(matrix[1, 0], matrix[0, 0])
    return out


def _box_corners_2d(box):
    """Four BEV corners of a box, (4,2)"""
    x, y, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2],
                      [dx / 2, dy / 2], [-dx / 2, dy / 2]])
    rot = np.array([[c, -s], [s, c]])
    return (rot @ local.T).T + np.array([x, y])


def bev_iou(box_a, box_b):
    """Rotated BEV IoU of two boxes"""
    import cv2

    rect_a = ((float(box_a[0]), float(box_a[1])),
              (float(box_a[3]), float(box_a[4])), float(np.degrees(box_a[6])))
    rect_b = ((float(box_b[0]), float(box_b[1])),
              (float(box_b[3]), float(box_b[4])), float(np.degrees(box_b[6])))

    ret, pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if ret == cv2.INTERSECT_NONE or pts is None:
        return 0.0

    inter = abs(cv2.contourArea(pts))
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    union = area_a + area_b - inter
    return float(np.clip(inter / union, 0.0, 1.0)) if union > 0 else 0.0


def _merge_pair(box_a, score_a, box_b, score_b):
    """Merge the boxes of two LiDARs for the same object, weighted by their scores."""
    wa, wb = float(score_a), float(score_b)
    total = wa + wb
    merged = (box_a * wa + box_b * wb) / total

    # headings are averaged on the circle, after resolving the +-pi ambiguity of the heading
    yaw_a, yaw_b = box_a[6], box_b[6]
    if abs(np.arctan2(np.sin(yaw_a - yaw_b), np.cos(yaw_a - yaw_b))) > np.pi / 2:
        yaw_b += np.pi
    merged[6] = np.arctan2(wa * np.sin(yaw_a) + wb * np.sin(yaw_b),
                           wa * np.cos(yaw_a) + wb * np.cos(yaw_b))
    return merged


def late_fuse(detections, matrices, iou_thresh=0.3):
    """Fuse the detections of several LiDARs in the world frame.

    :param detections: list of dict, output of PretrainedDetector.infer,
                       each with 'boxes'/'scores'/'labels'/'project_labels'
    :param matrices: list of 4x4 matrices from each LiDAR frame to the world frame
    :param iou_thresh: BEV IoU threshold for two LiDARs seeing the same object
    :return: dict with
        'boxes'          (K,7) world frame
        'scores'         (K,)  fused score
        'labels'         (K,)  nuScenes class names
        'project_labels' (K,)
        'num_sensors'    (K,)  number of LiDARs that detected the object (cross-sensor agreement, a quality signal)
        'agreement'      (K,)  BEV IoU of the two boxes when paired, 0 when single
    """
    world = []
    for det, mat in zip(detections, matrices):
        world.append({
            'boxes': transform_boxes(det['boxes'], mat),
            'scores': det['scores'],
            'labels': det['labels'],
            'project_labels': det['project_labels'],
        })

    if len(world) == 1:
        d = world[0]
        n = len(d['boxes'])
        return {**d, 'num_sensors': np.ones(n, np.int32), 'agreement': np.zeros(n, np.float32)}

    a, b = world[0], world[1]
    na, nb = len(a['boxes']), len(b['boxes'])

    # match only within a class, so a pedestrian and a bicycle are never merged
    pairs = []
    for i in range(na):
        for j in range(nb):
            if a['project_labels'][i] != b['project_labels'][j]:
                continue
            iou = bev_iou(a['boxes'][i], b['boxes'][j])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))

    # greedy pairing by decreasing IoU, each box used once
    pairs.sort(reverse=True)
    used_a, used_b = set(), set()
    matched = []
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((iou, i, j))

    boxes, scores, labels, plabels, nsens, agree = [], [], [], [], [], []

    for iou, i, j in matched:
        sa, sb = a['scores'][i], b['scores'][j]
        boxes.append(_merge_pair(a['boxes'][i], sa, b['boxes'][j], sb))
        # detected independently by two LiDARs -> combine the scores as independent evidence
        scores.append(1.0 - (1.0 - sa) * (1.0 - sb))
        labels.append(a['labels'][i] if sa >= sb else b['labels'][j])
        plabels.append(a['project_labels'][i])
        nsens.append(2)
        agree.append(iou)

    for src, used in ((a, used_a), (b, used_b)):
        for i in range(len(src['boxes'])):
            if i in used:
                continue
            boxes.append(src['boxes'][i])
            scores.append(float(src['scores'][i]))
            labels.append(src['labels'][i])
            plabels.append(int(src['project_labels'][i]))
            nsens.append(1)
            agree.append(0.0)

    if not boxes:
        return {'boxes': np.zeros((0, 7), np.float32), 'scores': np.zeros(0, np.float32),
                'labels': np.array([], dtype=object), 'project_labels': np.zeros(0, np.int32),
                'num_sensors': np.zeros(0, np.int32), 'agreement': np.zeros(0, np.float32)}

    order = np.argsort(scores)[::-1]
    return {
        'boxes': np.asarray(boxes, np.float32)[order],
        'scores': np.asarray(scores, np.float32)[order],
        'labels': np.asarray(labels, dtype=object)[order],
        'project_labels': np.asarray(plabels, np.int32)[order],
        'num_sensors': np.asarray(nsens, np.int32)[order],
        'agreement': np.asarray(agree, np.float32)[order],
    }


def assign_points_to_boxes(points_xyz, boxes):
    """Label each point with the index of its box; points outside every box get -1.

    Three filters: union AABB of all boxes -> KD-tree bounding-sphere query -> exact in-box test,
    to avoid building a tree on, or scanning O(points x boxes) over, the full cloud.

    :param points_xyz: (N,3)
    :param boxes: (K,7) world frame, in decreasing score order (overlapping boxes: the higher score takes the point)
    :return: (N,) int32, values in [-1, K-1]
    """
    cluster_id = np.full(len(points_xyz), -1, dtype=np.int32)
    if len(boxes) == 0 or len(points_xyz) == 0:
        return cluster_id

    # coarse filter with the union AABB of all boxes. Usually only a few percent of the ~260k points
    # lie near a box; building a KD-tree on all points is the main cost.
    radii = 0.5 * np.linalg.norm(boxes[:, 3:6], axis=1)
    lo = (boxes[:, :3] - radii[:, None]).min(axis=0)
    hi = (boxes[:, :3] + radii[:, None]).max(axis=0)
    near = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)
    idx_map = np.flatnonzero(near)
    if idx_map.size == 0:
        return cluster_id
    subset = points_xyz[idx_map]

    tree = cKDTree(subset)

    for k, box in enumerate(boxes):
        x, y, z, dx, dy, dz, yaw = box[:7]
        radius = 0.5 * float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        cand = np.asarray(tree.query_ball_point([x, y, z], radius), dtype=np.int64)
        if cand.size == 0:
            continue

        d = subset[cand] - np.array([x, y, z])
        c, s = np.cos(-yaw), np.sin(-yaw)
        lx = d[:, 0] * c - d[:, 1] * s
        ly = d[:, 0] * s + d[:, 1] * c
        inside = (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) & (np.abs(d[:, 2]) <= dz / 2)

        # cand indexes subset; map back to indices of the original cloud
        take = idx_map[cand[inside]]
        # with overlapping boxes the first written wins; passing boxes in decreasing score order lets the higher score win
        cluster_id[take] = np.where(cluster_id[take] < 0, k, cluster_id[take])

    return cluster_id


# minimum points inside a box, per class. Pedestrians and two-wheelers have few points: an OS-1-128
# sees roughly 14 points on a pedestrian at 40 m, so a single high threshold would remove distant pedestrians.
DEFAULT_MIN_POINTS = {
    'pedestrian': 5,
    'bicycle': 5,
    'motorcycle': 5,
    'car': 20,
    'truck': 20,
    'bus': 20,
    'trailer': 20,
    'construction_vehicle': 20,
}


def filter_by_point_count(fused, cluster_id, min_points=5):
    """Drop detections with too few points inside the box (removes most hallucinated boxes in sparse far areas).

    :param min_points: int (all classes) or dict (class name -> threshold, 5 for other classes)
    :return: (filtered_fused, remapped_cluster_id, counts)
    """
    n_boxes = len(fused['boxes'])
    if n_boxes == 0:
        return fused, cluster_id, np.zeros(0, np.int32)

    counts = np.bincount(cluster_id[cluster_id >= 0], minlength=n_boxes).astype(np.int32)

    if isinstance(min_points, dict):
        thresholds = np.array([min_points.get(str(l), 5) for l in fused['labels']], dtype=np.int32)
    else:
        thresholds = np.full(n_boxes, int(min_points), dtype=np.int32)
    keep = counts >= thresholds

    remap = np.full(n_boxes, -1, dtype=np.int32)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)

    new_cluster = np.where(cluster_id >= 0, remap[np.clip(cluster_id, 0, None)], -1).astype(np.int32)
    new_cluster[cluster_id < 0] = -1

    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n_boxes else v)
           for k, v in fused.items()}
    return out, new_cluster, counts[keep]