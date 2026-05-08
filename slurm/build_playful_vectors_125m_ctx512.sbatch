#!/bin/bash
#SBATCH --job-name=build-125m-playful
#SBATCH --partition=gpu
#SBATCH --output=build_playful_vectors_125m_ctx512_%j.out
#SBATCH --error=build_playful_vectors_125m_ctx512_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_125m_full_ctx512.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_125m_full_ctx512/best.pt}"
OUTPUT="${OUTPUT:-benchmarks/results/playful_serious_vectors_125m_full_ctx512.pt}"
LAYERS="${LAYERS:-all}"
PROMPT_BANK="${PROMPT_BANK:-prompt_banks/playful_vs_serious.yaml}"
POOLING="${POOLING:-mean}"

echo "Resolved build vector settings:"
printf '  CONFIG=%q\n' "$CONFIG"
printf '  CHECKPOINT=%q\n' "$CHECKPOINT"
printf '  OUTPUT=%q\n' "$OUTPUT"
printf '  LAYERS=%q\n' "$LAYERS"
printf '  PROMPT_BANK=%q\n' "$PROMPT_BANK"
printf '  POOLING=%q\n' "$POOLING"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT"
  exit 1
fi

python -u scripts/05_build_emotion_vectors.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --layers "$LAYERS" \
  --prompt-bank "$PROMPT_BANK" \
  --pooling "$POOLING"
