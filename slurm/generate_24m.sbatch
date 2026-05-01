#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=generate_24m_%j.out
#SBATCH --error=generate_24m_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_24m.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_24m/ckpt.pt}"
PROMPT="${PROMPT:-Once upon a time}"

python scripts/03_generate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT"
