#!/bin/bash
#SBATCH --job-name=activation-125m-ft
#SBATCH --partition=gpu
#SBATCH --output=train_125m_full_ctx512_continue_%j.out
#SBATCH --error=train_125m_full_ctx512_continue_%j.err
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

CONFIG="configs/tinystories_125m_full_ctx512_continue.yaml"
SOURCE_CKPT="checkpoints/tinystories_125m_full_ctx512/best.pt"
CKPT_DIR="checkpoints/tinystories_125m_full_ctx512_continue"
TARGET_CKPT="$CKPT_DIR/ckpt.pt"
TARGET_BEST="$CKPT_DIR/best.pt"

if [ ! -f data/processed/tinystories_full/train.npy ]; then
  echo "Missing data/processed/tinystories_full/train.npy"
  echo "Run sbatch slurm/prepare_tinystories_full.sbatch first."
  exit 1
fi

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Missing source checkpoint: $SOURCE_CKPT"
  exit 1
fi

mkdir -p "$CKPT_DIR"
if [ ! -f "$TARGET_CKPT" ]; then
  cp "$SOURCE_CKPT" "$TARGET_CKPT"
  cp "$SOURCE_CKPT" "$TARGET_BEST"
  echo "initialized_continue_from=$SOURCE_CKPT"
else
  echo "resuming_existing_continue_checkpoint=$TARGET_CKPT"
fi

python -u scripts/02_train.py --config "$CONFIG" --resume
