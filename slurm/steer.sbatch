#!/bin/bash
#SBATCH --job-name=steer
#SBATCH --partition=gpu
#SBATCH --output=steer_%j.out
#SBATCH --error=steer_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH="${REQUIRE_TORCH:-1}"
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_100m_full.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_100m_full/best.pt}"
VECTORS="${VECTORS:-benchmarks/results/emotion_vectors_100m_full.pt}"
EMOTION="${EMOTION:-scared}"
LAYER="${LAYER:-7}"
ALPHA="${ALPHA:-2.0}"
POSITION="${POSITION:-all}"
PROMPT="${PROMPT:-Once upon a time there was a little robot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"
STOP_AT_EOS="${STOP_AT_EOS:-1}"

stop_args=()
if [[ "$STOP_AT_EOS" == "1" || "$STOP_AT_EOS" == "true" || "$STOP_AT_EOS" == "TRUE" ]]; then
  stop_args+=(--stop-at-eos)
else
  stop_args+=(--no-stop-at-eos)
fi

echo "Resolved steer settings:"
printf '  CONFIG=%q\n' "$CONFIG"
printf '  CHECKPOINT=%q\n' "$CHECKPOINT"
printf '  VECTORS=%q\n' "$VECTORS"
printf '  EMOTION=%q\n' "$EMOTION"
printf '  LAYER=%q\n' "$LAYER"
printf '  ALPHA=%q\n' "$ALPHA"
printf '  POSITION=%q\n' "$POSITION"
printf '  PROMPT=%q\n' "$PROMPT"
printf '  MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
printf '  TEMPERATURE=%q\n' "$TEMPERATURE"
printf '  TOP_K=%q\n' "$TOP_K"
printf '  STOP_AT_EOS=%q\n' "$STOP_AT_EOS"

python -u scripts/06_steer_generation.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --vectors "$VECTORS" \
  --emotion "$EMOTION" \
  --layer "$LAYER" \
  --alpha "$ALPHA" \
  --position "$POSITION" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  "${stop_args[@]}"
