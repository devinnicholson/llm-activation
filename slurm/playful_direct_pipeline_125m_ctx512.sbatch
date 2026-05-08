#!/bin/bash
#SBATCH --job-name=playful-125m-direct
#SBATCH --partition=gpu
#SBATCH --output=playful_direct_125m_ctx512_%j.out
#SBATCH --error=playful_direct_125m_ctx512_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_125m_full_ctx512.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_125m_full_ctx512/best.pt}"
PROMPT_BANK="${PROMPT_BANK:-prompt_banks/playful_vs_serious_direct.yaml}"
VECTORS="${VECTORS:-benchmarks/results/playful_direct_vectors_125m_full_ctx512.pt}"
OUTPUT="${OUTPUT:-benchmarks/results/playful_direct_125m_ctx512_${SLURM_JOB_ID}.jsonl}"
LAYERS="${LAYERS:-3,4,5,6,7,8}"
ALPHAS="${ALPHAS:-0.5,1,1.5,2,3,5,7.5,10}"
POSITIONS="${POSITIONS:-last,all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"
POOLING="${POOLING:-mean}"

echo "Resolved 125M direct playful settings:"
printf '  CONFIG=%q\n' "$CONFIG"
printf '  CHECKPOINT=%q\n' "$CHECKPOINT"
printf '  PROMPT_BANK=%q\n' "$PROMPT_BANK"
printf '  VECTORS=%q\n' "$VECTORS"
printf '  OUTPUT=%q\n' "$OUTPUT"
printf '  LAYERS=%q\n' "$LAYERS"
printf '  ALPHAS=%q\n' "$ALPHAS"
printf '  POSITIONS=%q\n' "$POSITIONS"
printf '  MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
printf '  TEMPERATURE=%q\n' "$TEMPERATURE"
printf '  TOP_K=%q\n' "$TOP_K"
printf '  POOLING=%q\n' "$POOLING"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT"
  exit 1
fi

python -u scripts/05_build_emotion_vectors.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$VECTORS" \
  --layers all \
  --prompt-bank "$PROMPT_BANK" \
  --pooling "$POOLING"

python -u scripts/07_steering_sweep.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --vectors "$VECTORS" \
  --output "$OUTPUT" \
  --emotions playful \
  --layers "$LAYERS" \
  --alphas "$ALPHAS" \
  --positions "$POSITIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --stop-at-eos

python -u scripts/08_score_steering_sweep.py --input "$OUTPUT"
python -u scripts/10_report_steering_sweep.py --input "$OUTPUT"
