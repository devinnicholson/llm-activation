#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=generate_%j.out
#SBATCH --error=generate_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tiny.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tiny/ckpt.pt}"
PROMPT="${PROMPT:-High performance computing}"

python scripts/03_generate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT"
