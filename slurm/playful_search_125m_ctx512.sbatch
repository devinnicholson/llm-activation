#!/bin/bash
#SBATCH --job-name=playful-125m-search
#SBATCH --partition=gpu
#SBATCH --output=playful_search_125m_ctx512_%j.out
#SBATCH --error=playful_search_125m_ctx512_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_125m_full_ctx512.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_125m_full_ctx512/best.pt}"
VECTORS="${VECTORS:-benchmarks/results/playful_serious_vectors_125m_full_ctx512.pt}"
OUTPUT="${OUTPUT:-benchmarks/results/playful_search_125m_ctx512_${SLURM_JOB_ID}.jsonl}"
EMOTIONS="${EMOTIONS:-playful}"
LAYERS="${LAYERS:-4,6,8,10,11}"
ALPHAS="${ALPHAS:-0.5,1,1.5,2,3,5}"
POSITIONS="${POSITIONS:-last,all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"

echo "Resolved steering search settings:"
printf '  CONFIG=%q\n' "$CONFIG"
printf '  CHECKPOINT=%q\n' "$CHECKPOINT"
printf '  VECTORS=%q\n' "$VECTORS"
printf '  OUTPUT=%q\n' "$OUTPUT"
printf '  EMOTIONS=%q\n' "$EMOTIONS"
printf '  LAYERS=%q\n' "$LAYERS"
printf '  ALPHAS=%q\n' "$ALPHAS"
printf '  POSITIONS=%q\n' "$POSITIONS"
printf '  MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
printf '  TEMPERATURE=%q\n' "$TEMPERATURE"
printf '  TOP_K=%q\n' "$TOP_K"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT"
  exit 1
fi

if [ ! -f "$VECTORS" ]; then
  echo "Missing vectors: $VECTORS"
  echo "Run sbatch slurm/build_playful_vectors_125m_ctx512.sbatch first."
  exit 1
fi

python -u scripts/07_steering_sweep.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --vectors "$VECTORS" \
  --output "$OUTPUT" \
  --emotions "$EMOTIONS" \
  --layers "$LAYERS" \
  --alphas "$ALPHAS" \
  --positions "$POSITIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --stop-at-eos

python -u scripts/08_score_steering_sweep.py --input "$OUTPUT"
python -u scripts/10_report_steering_sweep.py --input "$OUTPUT"
