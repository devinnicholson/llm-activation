#!/bin/bash
#SBATCH --job-name=scratch-llm-ddp
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
source slurm/cluster_env.sh

torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node=4 \
  scripts/02_train.py \
  --config configs/base_125m.yaml
