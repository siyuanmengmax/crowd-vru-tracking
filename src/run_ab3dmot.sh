#!/bin/bash
# AB3DMOT on the VRU boxes (label 0) of an input tag, with track management <max_age> <min_hits>.
# Writes tag <src>_a<A>h<H> (symlinks to the input's boxes, members, and point bins; the source tag is not touched),
# or, with A=stock, runs the official Pedestrian setting (4 frames, 1 hit) in the source tag itself.
# usage: run_ab3dmot.sh <src_tag> <max_age|stock> <min_hits> <scenario>...
# needs AB3DMOT_ROOT (official clone) and XINSHUO_ROOT (Xinshuo_PyToolbox) and the AB3DMOT Python environment.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
: "${AB3DMOT_ROOT:=$HERE/../third_party/AB3DMOT}"; : "${XINSHUO_ROOT:=$HERE/../third_party/Xinshuo_PyToolbox}"
export PYTHONPATH="$AB3DMOT_ROOT:$XINSHUO_ROOT:$PYTHONPATH"
WORK=${WORK:-$HERE/work_vru}
SRC=$1; A=$2; H=$3; shift 3
cd "$AB3DMOT_ROOT"
for SEQ in "$@"; do
  if [ "$A" = stock ]; then TAG=$SRC; D=$WORK/$SEQ/$TAG; ENVS=""
  else TAG=${SRC}_a${A}h${H}; D=$WORK/$SEQ/$TAG; mkdir -p $D; ENVS="AB3D_MAXAGE=$A AB3D_MINHITS=$H"
       for f in detection_bbox_results.csv members.pkl detection; do ln -sfn $WORK/$SEQ/$SRC/$f $D/$f; done; fi
  DET=vru_${SEQ}_${TAG}
  python3 $HERE/ab3dmot/convert_roadside_lidar.py $D/detection_bbox_results.csv --det_name $DET --split val \
      --label_map "0:1" --coord_mode roadside --output_dir ./data/ROADSIDE/detection > /dev/null
  mkdir -p ./results/ROADSIDE/${DET}_Car_val_H1 ./results/ROADSIDE/${DET}_Pedestrian_val_H1   # main.py does not create them
  env $ENVS python3 $HERE/ab3dmot/run_roadside.py --dataset ROADSIDE --split val --det_name $DET > /dev/null 2>&1 || true   # combine_trk_cat fails after the Pedestrian results are written
  python3 $HERE/ab3dmot/convert_tracking_results.py --input_dir ./results/ROADSIDE/${DET}_Pedestrian_val_H1/data_0 \
      --output_file $D/AB3DMOT_matched.csv --reverse_label_map "1:0" --coord_transform roadside > /dev/null
done
