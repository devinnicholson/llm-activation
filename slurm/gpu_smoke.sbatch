#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=gpu_smoke_%j.out
#SBATCH --error=gpu_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
source slurm/cluster_env.sh

echo "GPU visibility"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(idx, torch.cuda.get_device_name(idx))
PY

python scripts/02_train.py --config configs/tiny.yaml
