#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=train_100m_full_%j.out
#SBATCH --error=train_100m_full_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_100m_full.yaml}"
RESUME="${RESUME:-0}"

if [ ! -f data/processed/tinystories_full/train.npy ]; then
  echo "Missing data/processed/tinystories_full/train.npy"
  echo "Run sbatch slurm/prepare_tinystories_full.sbatch first."
  exit 1
fi

if [ "$RESUME" = "1" ]; then
  python -u scripts/02_train.py --config "$CONFIG" --resume
else
  python -u scripts/02_train.py --config "$CONFIG"
fi
