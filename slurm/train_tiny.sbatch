#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=tiny_%j.out
#SBATCH --error=tiny_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh
RESUME="${RESUME:-0}"

python scripts/00_train_tokenizer.py --config configs/tiny.yaml
python scripts/01_prepare_dataset.py --config configs/tiny.yaml
if [ "$RESUME" = "1" ]; then
  python scripts/02_train.py --config configs/tiny.yaml --resume
else
  python scripts/02_train.py --config configs/tiny.yaml
fi
