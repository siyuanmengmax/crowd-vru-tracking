#!/bin/bash
# Self-training baseline (protocol in the paper, Self-Training Baseline): VoxelNeXt fine-tuned on zero-shot pseudo-labels of
# unlabeled LUMPI M6 frames, with ft_train.py unchanged (8 epochs). Usage: bash setup.sh && bash run_st.sh
cd "$(dirname "$0")"; P=$(pwd); mkdir -p out logs
export OPENPCDET=$P/OpenPCDet FT_DATA=$P/ft_data CKPT_DIR=$P/ckpt OUT_DIR=$P/out
python -u st_label.py > logs/st_label.log 2>&1 || { echo "teacher check or labeling failed:"; tail -5 logs/st_label.log; exit 1; }
tail -2 logs/st_label.log
ARCH=voxelnext python -u ../ft_train.py > logs/st_voxelnext.log 2>&1; echo "train exit $?"; tail -3 logs/st_voxelnext.log
python -u st_infer.py > logs/st_infer.log 2>&1; tail -2 logs/st_infer.log
python gpu_stress.py > logs/gpu_stress_after.log 2>&1; tail -1 logs/gpu_stress_after.log
tar czf results_st.tgz out/ft_ckpt_voxelnext/ep8.pth out/ft_ckpt_voxelnext/train.log ft_data/infos_train.pkl st_eval_out logs && echo "ALL DONE -> results_st.tgz"
