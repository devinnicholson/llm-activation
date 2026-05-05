#!/bin/bash
#SBATCH --job-name=activation-100m-ctx512
#SBATCH --partition=gpu
#SBATCH --output=train_100m_full_ctx512_%j.out
#SBATCH --error=train_100m_full_ctx512_%j.err
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

CONFIG="configs/tinystories_100m_full_ctx512.yaml"
SOURCE_CKPT="checkpoints/tinystories_100m_full/best.pt"
CKPT_DIR="checkpoints/tinystories_100m_full_ctx512"
TARGET_CKPT="${CKPT_DIR}/ckpt.pt"

if [ ! -f data/processed/tinystories_full/train.npy ]; then
  echo "Missing data/processed/tinystories_full/train.npy"
  echo "Run sbatch slurm/prepare_tinystories_full.sbatch first."
  exit 1
fi

if [ ! -f "$TARGET_CKPT" ]; then
  if [ ! -f "$SOURCE_CKPT" ]; then
    echo "Missing $SOURCE_CKPT"
    echo "Download or train the 100m_full best checkpoint first."
    exit 1
  fi
  mkdir -p "$CKPT_DIR"
  cp "$SOURCE_CKPT" "$TARGET_CKPT"
  echo "initialized_ctx512_from=$SOURCE_CKPT"
else
  echo "resuming_existing_ctx512_checkpoint=$TARGET_CKPT"
fi

python -u scripts/02_train.py --config "$CONFIG" --resume
