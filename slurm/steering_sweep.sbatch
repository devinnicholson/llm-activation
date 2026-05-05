#!/bin/bash
#SBATCH --job-name=steering-sweep
#SBATCH --partition=gpu
#SBATCH --output=steering_sweep_%j.out
#SBATCH --error=steering_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH="${REQUIRE_TORCH:-1}"
source slurm/cluster_env.sh

JOB_ID="${SLURM_JOB_ID:-manual}"
CONFIG="${CONFIG:-configs/tinystories_100m_full.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_100m_full/best.pt}"
VECTORS="${VECTORS:-benchmarks/results/emotion_vectors_100m_full.pt}"
OUTPUT="${OUTPUT:-benchmarks/results/steering_sweep_${JOB_ID}.jsonl}"
EMOTIONS="${EMOTIONS:-happy,sad,scared,calm,curious}"
LAYERS="${LAYERS:-4,6,7,9}"
ALPHAS="${ALPHAS:-1,2,3}"
POSITIONS="${POSITIONS:-last,all}"
PROMPT="${PROMPT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"
STOP_AT_EOS="${STOP_AT_EOS:-1}"
OVERWRITE="${OVERWRITE:-0}"
REPORT="${REPORT:-1}"

prompt_args=()
prompt_display="<script defaults>"
if [[ -n "$PROMPT" ]]; then
  prompt_args+=(--prompt "$PROMPT")
  prompt_display="$PROMPT"
fi
stop_args=()
if [[ "$STOP_AT_EOS" == "1" || "$STOP_AT_EOS" == "true" || "$STOP_AT_EOS" == "TRUE" ]]; then
  stop_args+=(--stop-at-eos)
else
  stop_args+=(--no-stop-at-eos)
fi
overwrite_args=()
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  overwrite_args+=(--overwrite)
fi

echo "Resolved steering sweep settings:"
printf '  CONFIG=%q\n' "$CONFIG"
printf '  CHECKPOINT=%q\n' "$CHECKPOINT"
printf '  VECTORS=%q\n' "$VECTORS"
printf '  OUTPUT=%q\n' "$OUTPUT"
printf '  EMOTIONS=%q\n' "$EMOTIONS"
printf '  LAYERS=%q\n' "$LAYERS"
printf '  ALPHAS=%q\n' "$ALPHAS"
printf '  POSITIONS=%q\n' "$POSITIONS"
printf '  PROMPT=%q\n' "$prompt_display"
printf '  MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
printf '  TEMPERATURE=%q\n' "$TEMPERATURE"
printf '  TOP_K=%q\n' "$TOP_K"
printf '  STOP_AT_EOS=%q\n' "$STOP_AT_EOS"
printf '  OVERWRITE=%q\n' "$OVERWRITE"
printf '  REPORT=%q\n' "$REPORT"

python -u scripts/07_steering_sweep.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --vectors "$VECTORS" \
  --output "$OUTPUT" \
  --emotions "$EMOTIONS" \
  --layers "$LAYERS" \
  --alphas "$ALPHAS" \
  --positions "$POSITIONS" \
  "${prompt_args[@]}" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  "${stop_args[@]}" \
  "${overwrite_args[@]}"

python -u scripts/08_score_steering_sweep.py --input "$OUTPUT"
if [[ "$REPORT" == "1" || "$REPORT" == "true" || "$REPORT" == "TRUE" ]]; then
  python -u scripts/10_report_steering_sweep.py --input "$OUTPUT"
fi
