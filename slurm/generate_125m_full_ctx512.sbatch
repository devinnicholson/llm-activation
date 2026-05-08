#!/bin/bash
#SBATCH --job-name=activation-125m-gen
#SBATCH --partition=gpu
#SBATCH --output=generate_125m_full_ctx512_%j.out
#SBATCH --error=generate_125m_full_ctx512_%j.err
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

CONFIG="${CONFIG:-configs/tinystories_125m_full_ctx512.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_125m_full_ctx512/best.pt}"
PROMPT="${PROMPT:-Once upon a time there was a little robot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-350}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"
STOP_AT_EOS="${STOP_AT_EOS:-1}"

echo "Resolved generate settings:"
echo "  CONFIG=$CONFIG"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  PROMPT=$PROMPT"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "  TEMPERATURE=$TEMPERATURE"
echo "  TOP_K=$TOP_K"
echo "  STOP_AT_EOS=$STOP_AT_EOS"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT"
  echo "Expected the finished 125M ctx512 run to have written best.pt."
  exit 1
fi

stop_args=()
if [[ "$STOP_AT_EOS" == "1" || "$STOP_AT_EOS" == "true" || "$STOP_AT_EOS" == "TRUE" ]]; then
  stop_args+=(--stop-at-eos)
else
  stop_args+=(--no-stop-at-eos)
fi

python -u scripts/03_generate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  "${stop_args[@]}"
