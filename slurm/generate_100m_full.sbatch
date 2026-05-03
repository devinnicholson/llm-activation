#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=generate_100m_full_%j.out
#SBATCH --error=generate_100m_full_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_100m_full.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_100m_full/best.pt}"
PROMPT="${PROMPT:-Once upon a time there was a little robot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"

python -u scripts/03_generate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K"
