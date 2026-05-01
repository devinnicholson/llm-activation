#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=train_60m_%j.out
#SBATCH --error=train_60m_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_60m_overnight.yaml}"
RESUME="${RESUME:-0}"

if [ ! -f data/processed/tinystories_24m/tokenizer.json ]; then
  echo "Missing data/processed/tinystories_24m/tokenizer.json"
  echo "Run slurm/train_24m.sbatch first, or copy processed tinystories_24m data back."
  exit 1
fi

if [ "$RESUME" = "1" ]; then
  python scripts/02_train.py --config "$CONFIG" --resume
else
  python scripts/02_train.py --config "$CONFIG"
fi
