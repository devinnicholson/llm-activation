#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=prepare_tinystories_full_%j.out
#SBATCH --error=prepare_tinystories_full_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=0
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_100m_full.yaml}"

python -u scripts/09_download_tinystories_full.py

if [ ! -f data/processed/tinystories_24m/tokenizer.json ]; then
  echo "Missing data/processed/tinystories_24m/tokenizer.json"
  echo "Run slurm/train_24m.sbatch first, or copy processed tinystories_24m data back."
  exit 1
fi

python -u scripts/01_prepare_dataset.py --config "$CONFIG"
