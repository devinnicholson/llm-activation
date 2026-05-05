#!/bin/bash
#SBATCH --job-name=build-vectors
#SBATCH --partition=gpu
#SBATCH --output=build_vectors_%j.out
#SBATCH --error=build_vectors_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH="${REQUIRE_TORCH:-1}"
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_100m_full.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_100m_full/best.pt}"
if [[ -n "${OUTPUT:-}" ]]; then
  OUTPUT="$OUTPUT"
elif [[ -n "${VECTORS:-}" ]]; then
  OUTPUT="$VECTORS"
else
  OUTPUT="benchmarks/results/emotion_vectors_100m_full.pt"
fi
VECTORS="$OUTPUT"
LAYERS="${LAYERS:-all}"
EMOTIONS="${EMOTIONS:-}"
PROMPT_BANK="${PROMPT_BANK:-}"
BASELINE="${BASELINE:-}"
POOLING="${POOLING:-mean}"
NO_NORMALIZE="${NO_NORMALIZE:-0}"

vector_args=()
emotions_display="<script default or prompt-bank targets>"
if [[ -n "$EMOTIONS" ]]; then
  vector_args+=(--emotions "$EMOTIONS")
  emotions_display="$EMOTIONS"
fi
if [[ -n "$PROMPT_BANK" ]]; then
  vector_args+=(--prompt-bank "$PROMPT_BANK")
fi
if [[ -n "$BASELINE" ]]; then
  vector_args+=(--baseline "$BASELINE")
fi
if [[ "$NO_NORMALIZE" == "1" || "$NO_NORMALIZE" == "true" || "$NO_NORMALIZE" == "TRUE" ]]; then
  vector_args+=(--no-normalize)
fi

echo "Resolved build vector settings:"
printf '  CONFIG=%q\n' "$CONFIG"
printf '  CHECKPOINT=%q\n' "$CHECKPOINT"
printf '  OUTPUT=%q\n' "$OUTPUT"
printf '  VECTORS=%q\n' "$VECTORS"
printf '  LAYERS=%q\n' "$LAYERS"
printf '  EMOTIONS=%q\n' "$emotions_display"
printf '  PROMPT_BANK=%q\n' "${PROMPT_BANK:-<none>}"
printf '  BASELINE=%q\n' "${BASELINE:-<script default>}"
printf '  POOLING=%q\n' "$POOLING"
printf '  NO_NORMALIZE=%q\n' "$NO_NORMALIZE"

python -u scripts/05_build_emotion_vectors.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --layers "$LAYERS" \
  --pooling "$POOLING" \
  "${vector_args[@]}"
