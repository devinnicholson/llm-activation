#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=steering_sweep_100m_full_%j.out
#SBATCH --error=steering_sweep_100m_full_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

EMOTIONS="${EMOTIONS:-happy,sad,scared,calm,curious}"
LAYERS="${LAYERS:-4,6,7,9}"
ALPHAS="${ALPHAS:-1,2,3}"
POSITIONS="${POSITIONS:-last,all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
OUTPUT="${OUTPUT:-benchmarks/results/steering_sweep_100m_full_${SLURM_JOB_ID}.jsonl}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-100}"
STOP_AT_EOS="${STOP_AT_EOS:-1}"
OVERWRITE="${OVERWRITE:-0}"
REPORT="${REPORT:-1}"

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

python -u scripts/07_steering_sweep.py \
  --config configs/tinystories_100m_full.yaml \
  --checkpoint checkpoints/tinystories_100m_full/best.pt \
  --vectors benchmarks/results/emotion_vectors_100m_full.pt \
  --output "$OUTPUT" \
  --emotions "$EMOTIONS" \
  --layers "$LAYERS" \
  --alphas "$ALPHAS" \
  --positions "$POSITIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  "${stop_args[@]}" \
  "${overwrite_args[@]}"

python -u scripts/08_score_steering_sweep.py --input "$OUTPUT"
if [[ "$REPORT" == "1" || "$REPORT" == "true" || "$REPORT" == "TRUE" ]]; then
  python -u scripts/10_report_steering_sweep.py --input "$OUTPUT"
fi
