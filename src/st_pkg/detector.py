"""Pretrained 3D detector wrapper (OpenPCDet).

Feeds the ground-levelled point cloud of each LiDAR to a pretrained model and returns boxes in that LiDAR's own frame.
Frame conventions: see the docstring of `PretrainedDetector.infer`.
"""

import os
import sys
from pathlib import Path

import numpy as np

# OpenPCDet repository (shared, not copied)
DEFAULT_OPENPCDET_ROOT = Path(os.environ.get("OPENPCDET", Path(__file__).resolve().parents[2] / "third_party" / "OpenPCDet"))

# nuScenes LiDAR mounting height above ground; the model's z regression was learned at this height
NUSCENES_LIDAR_HEIGHT = 1.84

# nuScenes 10 classes -> project labels
# 0=person 1=bicycle 2=small vehicle 3=large vehicle 4=other
NUSCENES_TO_PROJECT = {
    'pedestrian': 0,
    'bicycle': 1,
    'motorcycle': 1,
    'car': 2,
    'truck': 3,
    'bus': 3,
    'trailer': 3,
    'construction_vehicle': 3,
    'barrier': 4,
    'traffic_cone': 4,
}

# classes that are meaningless and false-positive prone at a roadside site
DEFAULT_DROP_CLASSES = ('barrier', 'traffic_cone')


def _install_numpy_aliases():
    """OpenPCDet 0.6 predates numpy 2 and still uses the removed aliases np.int / np.float etc."""
    for alias, target in (('int', int), ('float', float), ('bool', bool), ('object', object)):
        if not hasattr(np, alias):
            setattr(np, alias, target)


class PretrainedDetector:
    """Batch inference wrapper for a pretrained OpenPCDet detector.

    Usage::

        det = PretrainedDetector(cfg_file, ckpt_path)
        results = det.infer([source_cloud, target_cloud])   # one forward pass, two LiDARs
    """

    def __init__(self, cfg_file, ckpt_path,
                 openpcdet_root=DEFAULT_OPENPCDET_ROOT,
                 z_sensor_offset=NUSCENES_LIDAR_HEIGHT,
                 score_thresh=0.45,
                 drop_classes=DEFAULT_DROP_CLASSES,
                 point_cloud_range=None,
                 device='cuda'):
        """
        :param cfg_file: config path relative to OpenPCDet tools/,
                         e.g. 'cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml'
        :param ckpt_path: .pth weights
        :param z_sensor_offset: how far (m) to lower the ground-levelled cloud so the ground lies where the model expects it;
                                1.84 for nuScenes models (their sensor height above ground).
        :param score_thresh: score threshold
        :param drop_classes: class names dropped directly
        :param point_cloud_range: overrides the detection range of the config [xmin,ymin,zmin,xmax,ymax,zmax]
        """
        self.openpcdet_root = Path(openpcdet_root)
        self.z_sensor_offset = float(z_sensor_offset)
        self.score_thresh = float(score_thresh)
        self.drop_classes = set(drop_classes or ())
        self.device = device

        if not self.openpcdet_root.exists():
            raise FileNotFoundError(f"OpenPCDet repository not found: {self.openpcdet_root}")
        if str(self.openpcdet_root) not in sys.path:
            sys.path.insert(0, str(self.openpcdet_root))

        _install_numpy_aliases()

        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        self.torch = torch

        # OpenPCDet yaml files use _BASE_CONFIG_ paths relative to tools/, so load from there
        tools_dir = self.openpcdet_root / 'tools'
        cwd = os.getcwd()
        os.chdir(tools_dir)
        try:
            cfg_from_yaml_file(str(cfg_file), cfg)
        finally:
            os.chdir(cwd)

        if point_cloud_range is not None:
            cfg.DATA_CONFIG.POINT_CLOUD_RANGE = list(point_cloud_range)

        self.cfg = cfg
        self.class_names = list(cfg.CLASS_NAMES)
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

        logger = common_utils.create_logger()
        self.dataset = _InMemoryDataset(cfg.DATA_CONFIG, self.class_names, logger)
        self.model = build_network(model_cfg=cfg.MODEL,
                                   num_class=len(self.class_names),
                                   dataset=self.dataset)
        self.model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
        self.model.to(device).eval()

        # model input features: [x,y,z,intensity,timestamp] for nuScenes models
        self.num_point_features = cfg.DATA_CONFIG.get('NUM_POINT_FEATURES', {}).get('use_dim', 5)
        if isinstance(self.num_point_features, (list, tuple)):
            self.num_point_features = len(self.num_point_features)

    def prepare_cloud(self, pcd_array, columns):
        """Convert a project point array to the model input format.

        :param pcd_array: (N, M) array, **already transformed by the LiDAR's ground_matrix**
                          (ground at z=0), but not yet into the world frame
        :param columns: column names, must contain 'x','y','z','range','reflectivity'
        :return: (N, num_point_features) float32

        Note that intensity is the **raw 0-255 reflectivity**, not normalised: nuScenes
        intensity is 0-255, and dividing by 255 disables the channel (measured: equivalent to all-zero input).
        """
        z_idx = columns.index('z')
        r_idx = columns.index('range')
        refl_idx = columns.index('reflectivity')

        valid = pcd_array[:, r_idx] > 0.5
        arr = pcd_array[valid]

        pts = np.zeros((len(arr), self.num_point_features), dtype=np.float32)
        pts[:, 0] = arr[:, columns.index('x')]
        pts[:, 1] = arr[:, columns.index('y')]
        pts[:, 2] = arr[:, z_idx] - self.z_sensor_offset
        if self.num_point_features > 3:
            pts[:, 3] = arr[:, refl_idx]
        # the 5th feature is the multi-sweep time lag in nuScenes; always 0 for single-frame inference
        return pts, valid

    def preprocess(self, clouds):
        """Voxelisation + batching. **CPU only**; measured at about 20% of the per-frame time.

        Exposed separately so that a producer thread can do it ahead of time, off the GPU critical path.

        :param clouds: list of (N, num_point_features) float32, each in **that LiDAR's
                       own ground-levelled frame** with z_sensor_offset already subtracted
        :return: batch dict (still on the CPU)
        """
        self.dataset.clouds = clouds
        return self.dataset.collate_batch([self.dataset[i] for i in range(len(clouds))])

    def infer_batch(self, batch):
        """Forward pass on the output of preprocess. Only this step uses the GPU.

        :return: see infer
        """
        torch = self.torch
        from pcdet.models import load_data_to_gpu

        load_data_to_gpu(batch)
        with torch.no_grad():
            preds, _ = self.model.forward(batch)

        names = np.array(self.class_names)
        out = []
        for d in preds:
            boxes = d['pred_boxes'].cpu().numpy()
            scores = d['pred_scores'].cpu().numpy()
            labels = names[d['pred_labels'].cpu().numpy() - 1]

            keep = scores >= self.score_thresh
            if self.drop_classes:
                keep &= ~np.isin(labels, list(self.drop_classes))
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(boxes):
                boxes = boxes.copy()
                boxes[:, 2] += self.z_sensor_offset   # back to the frame with the ground at z=0

            out.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'project_labels': np.array([NUSCENES_TO_PROJECT.get(c, 4) for c in labels],
                                           dtype=np.int32),
            })
        return out

    def infer(self, clouds):
        """Convenience wrapper of preprocess + infer_batch (CPU and GPU in series, for single-frame debugging).

        :param clouds: list of (N, num_point_features) float32, each in **that LiDAR's
                       own ground-levelled frame** with z_sensor_offset already subtracted
        :return: list of dict, each with
                 'boxes'  (K,7) [x,y,z,dx,dy,dz,heading], z_sensor_offset added back to z,
                                i.e. in the LiDAR's ground-levelled frame (ground at z=0)
                 'scores' (K,)
                 'labels' (K,) nuScenes class names
                 'project_labels' (K,) integer project labels
        """
        return self.infer_batch(self.preprocess(clouds))


class _InMemoryDataset:
    """Minimal DatasetTemplate subclass that takes point arrays from memory.

    Inherits DatasetTemplate only at run time (OpenPCDet must be on sys.path first).
    """

    def __new__(cls, dataset_cfg, class_names, logger=None):
        from pcdet.datasets import DatasetTemplate

        class _Impl(DatasetTemplate):
            def __init__(self, dataset_cfg, class_names, logger):
                super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                                 training=False, root_path=Path('.'), logger=logger)
                self.clouds = []

            def __len__(self):
                return len(self.clouds)

            def __getitem__(self, index):
                return self.prepare_data({'points': self.clouds[index],
                                          'frame_id': str(index)})

        return _Impl(dataset_cfg, class_names, logger)
