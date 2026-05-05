#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=steer_60m_%j.out
#SBATCH --error=steer_60m_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_60m_overnight.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_60m_overnight/ckpt.pt}"
VECTORS="${VECTORS:-benchmarks/results/emotion_vectors_60m.pt}"
EMOTION="${EMOTION:-happy}"
LAYER="${LAYER:--1}"
ALPHA="${ALPHA:-2.0}"
POSITION="${POSITION:-last}"
PROMPT="${PROMPT:-Once upon a time there was a little robot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-260}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"
STOP_AT_EOS="${STOP_AT_EOS:-1}"

stop_args=()
if [[ "$STOP_AT_EOS" == "1" || "$STOP_AT_EOS" == "true" || "$STOP_AT_EOS" == "TRUE" ]]; then
  stop_args+=(--stop-at-eos)
else
  stop_args+=(--no-stop-at-eos)
fi

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
