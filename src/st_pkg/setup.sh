#!/bin/bash
# One-time setup on a rented GPU (an image with PyTorch 2.x and the CUDA toolkit / nvcc, e.g. a "devel" template).
# Usage: bash setup.sh      (run from the unpacked package directory)
set -e
cd "$(dirname "$0")"; P=$(pwd)
python - <<'PY'
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
echo "== GPU correctness check (90 s)"; python gpu_stress.py | tee stress.log
if grep -q UNSTABLE stress.log; then echo "THIS GPU IS UNSTABLE - stop the pod and rent another one"; exit 1; fi
TV=$(python -c "import torch; print(torch.__version__.split('+')[0])"); CU=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
CUMAJ=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
pip install -q "numpy<2" pandas scipy easydict pyyaml numba tensorboardX scikit-image tqdm SharedArray opencv-python-headless pyquaternion
if [ "$CUMAJ" = "11" ]; then pip install -q spconv-cu118; else pip install -q spconv-cu120; fi
pip install -q torch_scatter -f "https://data.pyg.org/whl/torch-${TV}+${CU}.html"
cd "$P/OpenPCDet" && python setup.py develop > "$P/build.log" 2>&1 || { echo "OpenPCDet build failed, see build.log"; tail -20 "$P/build.log"; exit 1; }
python -c "import pcdet, spconv, torch_scatter; print('pcdet', pcdet.__version__, 'ok')"
echo "SETUP DONE"
