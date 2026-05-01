#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=train_8m_%j.out
#SBATCH --error=train_8m_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_8m.yaml}"
REBUILD_DATA="${REBUILD_DATA:-1}"

if [ "$REBUILD_DATA" = "1" ]; then
  python scripts/00_train_tokenizer.py --config "$CONFIG"
  python scripts/01_prepare_dataset.py --config "$CONFIG"
else
  echo "Skipping tokenizer and dataset rebuild because REBUILD_DATA=$REBUILD_DATA"
fi

python scripts/02_train.py --config "$CONFIG"
