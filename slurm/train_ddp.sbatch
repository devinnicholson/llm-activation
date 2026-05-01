#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=ddp_%j.out
#SBATCH --error=ddp_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh
RESUME="${RESUME:-0}"
TRAIN_ARGS=(--config configs/base_125m.yaml)

if [ "$RESUME" = "1" ]; then
  TRAIN_ARGS+=(--resume)
fi

torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node=4 \
  scripts/02_train.py \
  "${TRAIN_ARGS[@]}"
