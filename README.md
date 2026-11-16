# crowd-vru-tracking

Code for the paper

> S. Meng, *Reducing Identity Switches of Crowded Vulnerable Road Users from Roadside LiDAR without Site-Specific Labels*, manuscript submitted to *Advanced Engineering Informatics*.

The method is a front end that sits between roadside LiDAR point clouds and an unchanged multi-object tracker.
Every output vulnerable road user (VRU) is a foreground cluster or a part of one. Merged clusters are split at
kernel density peaks, and a part becomes a VRU when a pretrained public detector (nuScenes VoxelNeXt, no
site labels) confirms it, with an object memory that carries the confirmation through detector misses.

The repository contains the front end, the baselines, the tracker wrappers, and the evaluation used for every
table of the paper. Datasets, third-party trackers, detector weights, and the ignore-region annotations are
obtained separately (below).

## Layout

```
src/
  prep_v2x_vru.py, bg_build.py, prep_real.py    V2X-Real: labels, background model, foreground clusters (real0.5)
  lumpi_prep.py                                 LUMPI: labels, background model, foreground clusters
  det_v2x.py, lumpi_det.py, ground_map.py       zero-shot VoxelNeXt inference (GPU)
  hybrid.py (+ mvp_split.py, split_prep.py,     the proposed front end (tag hyb_<NAME>)
    bbox_fast.py, relabel_size.py)
  make_detonly.py, det_bins.py, det_fgfilter.py detector-only inputs and the foreground filter (det0.45f, ftvxn0.45f)
  relabel_size.py                               size-rule baseline (real0.5s)
  ft_prep.py, ft_train.py, det_ft.py            detector fine-tuned on V2X-Real labels
  st_prep.py, st_prep_cpu.py, st_prep_eval.py,  self-training baseline (pseudo-labels on unlabeled LUMPI M6 frames)
    st_pkg/                                     cloud package used for teacher labelling, training, and inference
  run_ab3dmot.sh, ab3dmot/                      AB3DMOT on roadside boxes (official code, unmodified, see below)
  run_immortal.py                               ImmortalTracker (official code, unmodified)
  run_trackers.py, tracking_single_point.py     Single Point UKF tracker
  eval_vru.py, eval_ign.py, lumpi_eval_ign.py,  CLEAR MOT / IDF1 with raw labels and with ignore regions
    lumpi_eval_ign_m6.py
  crowd_ids.py, assoc_eval.py, assoc_ci.py      crowd strata, AssA_m, Cover80, paired bootstrap
  dev_split.py, m6/                             training-split selection, held-out M6 clips
  xwalk/                                        crosswalk check
  v2x_prep.py, frgb3d_utils.py, compute_bbox.py helpers (reliability model and box fitting from FRGB3D)
```

## Requirements

Python 3.11 with numpy 1.26, pandas 2.2, scipy 1.14, scikit-learn 1.5, numba 0.60, filterpy 1.4.5, PyYAML,
tqdm, matplotlib, and opencv-python (`pip install -r requirements.txt`). Detector inference and training also
need PyTorch with CUDA, spconv, and OpenPCDet 0.6.0.

Third-party code, cloned into `third_party/` (or pointed to with the environment variables):

| Code | Source | Variable | Licence |
|---|---|---|---|
| AB3DMOT | https://github.com/xinshuoweng/AB3DMOT (commit 61f3bd7) | `AB3DMOT_ROOT` | CMU non-commercial research licence |
| Xinshuo_PyToolbox | https://github.com/xinshuoweng/Xinshuo_PyToolbox (commit 1918b9e) | `XINSHUO_ROOT` | see repository |
| ImmortalTracker | https://github.com/esdolo/ImmortalTracker (commit 11f8855) | `IMMORTAL_ROOT` | see repository |
| OpenPCDet | https://github.com/open-mmlab/OpenPCDet (v0.6.0) | `OPENPCDET` | Apache-2.0 |

None of these is modified or redistributed here. `src/ab3dmot/run_roadside.py` registers a roadside dataset in
AB3DMOT at run time (static sensor, no camera; Pedestrian parameters as the official KITTI Pedestrian setting,
track management from `AB3D_MAXAGE` / `AB3D_MINHITS`). The AB3DMOT licence restricts use to non-commercial
research; the MIT licence of this repository covers only the code in this repository.

The detector weights are the OpenPCDet nuScenes VoxelNeXt model `cbgs_voxel0075_voxelnext_nds6665.pth`
(set `VOXELNEXT_CKPT`).

## Data

- V2X-Real (Xiang et al., ECCV 2024), infrastructure LiDARs: set `V2X_ROOT` to the folder with the scenario
  directories.
- LUMPI (Busch et al., IEEE IV 2022), Measurements 4 and 6: set `LSRC` to the measurement folder and `LLABEL`
  to its `Label.csv`; for Measurement 6 also `LSENS` and `LCACHE=_m6` as in the commands below.
- Ignore-region annotations (review decisions for the candidate tracks of every method): [Zenodo DOI to be added];
  unpack into `annotations/` or set `ANNOT_DIR`.

## Reproducing the paper

All commands run from `src/`. `WORK` selects the work directory of a site; outputs of the evaluation go to
`results/` (or `RESULTS`).

```bash
# V2X-Real
python bg_build.py                                   # background model of the two infrastructure LiDARs
python prep_v2x_vru.py                               # labels (gt_all.pkl) per scenario
python prep_real.py                                  # foreground clusters, tag real0.5
python det_v2x.py                                    # zero-shot VoxelNeXt boxes (GPU)
KDE_TAU=0.20 DET_T=0.45 NAME=kde python hybrid.py    # proposed front end, tag hyb_kde
./run_ab3dmot.sh hyb_kde 20 3 <scenario>...          # AB3DMOT, 20 frames / 3 hits -> hyb_kde_a20h3
MINHITS=3 python run_immortal.py hyb_kde <scenario>... # ImmortalTracker -> hyb_kde_im3
TAGS=hyb_kde python run_trackers.py                  # Single Point UKF
TAGS=hyb_kde_a20h3,hyb_kde_im3,hyb_kde python eval_ign.py

# LUMPI Measurement 4 (six clips)
WORK=work_lumpi LWORK=work_lumpi python lumpi_prep.py
WORK=work_lumpi LWORK=work_lumpi python lumpi_det.py
WORK=work_lumpi KDE_TAU=0.20 DET_T=0.45 NAME=kde python hybrid.py
# ... trackers as above with WORK=work_lumpi, then
WORK=work_lumpi TAGS=hyb_kde_a20h3,hyb_kde_im3,hyb_kde python lumpi_eval_ign.py
```

Front-end variants of the ablation are options of `hybrid.py`: `MEM=0` (no memory confirmation), `DEDUP=0.8`
without `KDE_TAU` (detector-based count), `DEDUP=0.8 SPLIT=0` (no splitting; `SPLIT=0` has no effect on the KDE
count), and the relaxations `RELAX=1`, `MEM=1000000`, `COAST=10`. The docstring of `hybrid.py` lists all options.
Baseline inputs: `make_detonly.py`, `det_bins.py`, `det_fgfilter.py` (tags `det0.45f`, `ftvxn0.45f`, `stvxn0.45f`)
and `relabel_size.py` (tag `real0.5s`). Track management of every input was selected on the V2X-Real training
splits with `dev_split.py`.

AssA_m and Cover80: `python assoc_eval.py` (with `WORK_V2X`, `WORK_M4`, `WORK_M6`) and `python assoc_ci.py`.

### Checks of this release

The released code was run against the outputs behind the paper on one V2X-Real test scenario
(background model, labels, clusters, front end, and all three trackers gave byte-identical files) and the
evaluation was rerun on all V2X-Real evaluation scenarios and M4 clips:

| Site | Tracker | MOTA | IDF1 | IDS |
|---|---|---|---|---|
| V2X-Real | AB3DMOT 20/3 | 66.9 | 79.2 | 508 |
| V2X-Real | ImmortalTracker | 71.9 | 81.7 | 717 |
| V2X-Real | Single Point UKF | 65.7 | 73.2 | 820 |
| LUMPI M4 | AB3DMOT 20/3 | 65.5 | 64.9 | 857 |
| LUMPI M4 | ImmortalTracker | 59.3 | 64.9 | 867 |
| LUMPI M4 | Single Point UKF | 45.6 | 48.8 | 1330 |

Detector inference was not rerun for this check.

## Citation

```bibtex
@unpublished{meng_crowd_vru,
  author = {Meng, Siyuan},
  title  = {Reducing Identity Switches of Crowded Vulnerable Road Users from Roadside {LiDAR} without Site-Specific Labels},
  note   = {Manuscript submitted to Advanced Engineering Informatics}
}
```

## Licence

MIT (see `LICENSE`) for the code in this repository. Third-party code and the datasets keep their own licences.
