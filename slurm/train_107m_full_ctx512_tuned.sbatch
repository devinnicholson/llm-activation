#!/bin/bash
#SBATCH --job-name=activation-107m-tuned
#SBATCH --partition=gpu
#SBATCH --output=train_107m_full_ctx512_tuned_%j.out
#SBATCH --error=train_107m_full_ctx512_tuned_%j.err
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

CONFIG="configs/tinystories_107m_full_ctx512_tuned.yaml"
CKPT="checkpoints/tinystories_107m_full_ctx512_tuned/ckpt.pt"

if [ ! -f data/processed/tinystories_full/train.npy ]; then
  echo "Missing data/processed/tinystories_full/train.npy"
  echo "Run sbatch slurm/prepare_tinystories_full.sbatch first."
  exit 1
fi

if [ -f "$CKPT" ]; then
  echo "resuming_existing_checkpoint=$CKPT"
  python -u scripts/02_train.py --config "$CONFIG" --resume
else
  echo "starting_new_checkpoint=$CKPT"
  python -u scripts/02_train.py --config "$CONFIG"
fi
