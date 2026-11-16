"""Run the unmodified official AB3DMOT (https://github.com/xinshuoweng/AB3DMOT) on roadside LiDAR boxes,
Pedestrian category only. No AB3DMOT file is changed; this script registers a ROADSIDE dataset at run time:
  - dataset: sequence '0000', split folder = split, data root ./data/ROADSIDE (relative to the AB3DMOT directory);
  - static sensor: no ego poses and no camera calibration are loaded (ego_com is false in ROADSIDE.yml);
  - frame range: 0-29,999, as there is no image folder (the behaviour of the runs in the paper);
  - Pedestrian parameters: greedy matching, GIoU-3D threshold -0.4, min_hits 1, max_age 4 (the official KITTI
    Pedestrian values), optionally replaced by AB3D_MAXAGE / AB3D_MINHITS from the environment.
Run from the AB3DMOT directory: python <this file> --dataset ROADSIDE --split val --det_name <name>"""
import os, sys, types
import numpy as np
sys.path.insert(0, os.getcwd())
import main as M
import AB3DMOT_libs.utils as U
from AB3DMOT_libs import model as MD
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ROADSIDE.yml")

_config = M.Config
def Config(path):
    cfg, show = _config(CFG if path.endswith("ROADSIDE.yml") else path); cfg.cat_list = ["Pedestrian"]; return cfg, show
M.Config = Config

_gss = U.get_subfolder_seq
def get_subfolder_seq(dataset, split):
    if dataset != "ROADSIDE": return _gss(dataset, split)
    return split, {0: "Car", 1: "Pedestrian", 3: "Cyclist"}, {"image": (1080, 1920), "lidar": (720, 1920)}, ["0000"], os.path.abspath("./data/ROADSIDE")
U.get_subfolder_seq = M.get_subfolder_seq = get_subfolder_seq

ARGS = M.parse_args()
U.load_oxts = lambda path: None          # static sensor
U.Calibration = lambda path: None        # no camera
_list = U.load_list_from_folder
def load_list_from_folder(folder, *a, **k):
    # No image folder for a roadside sensor: frames 0-29,999, as in the experiments of the paper (tracks are written
    # for up to max_age - 1 frames after the last detection; frames without labels do not enter the evaluation).
    if os.path.exists(folder): return _list(folder, *a, **k)
    frames = [f"{i:06d}.txt" for i in range(30000)]
    return frames, len(frames)
U.load_list_from_folder = load_list_from_folder

_gp = MD.AB3DMOT.get_param
def get_param(self, cfg, cat):
    if cfg.dataset == "ROADSIDE":
        _gp(self, types.SimpleNamespace(dataset="KITTI", det_name="pvrcnn"), cat)   # KITTI Pedestrian: greedy, giou_3d, -0.4, 1, 4
    else: _gp(self, cfg, cat)
    if "AB3D_MAXAGE" in os.environ: self.max_age = int(os.environ["AB3D_MAXAGE"])
    if "AB3D_MINHITS" in os.environ: self.min_hits = int(os.environ["AB3D_MINHITS"])
MD.AB3DMOT.get_param = get_param

M.main(ARGS)
