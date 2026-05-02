#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=steering_sweep_60m_%j.out
#SBATCH --error=steering_sweep_60m_%j.err
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
LAYERS="${LAYERS:-6,9,11}"
ALPHAS="${ALPHAS:-2,3}"
POSITIONS="${POSITIONS:-last,all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
OUTPUT="${OUTPUT:-benchmarks/results/steering_sweep_60m_${SLURM_JOB_ID}.jsonl}"

python -u scripts/07_steering_sweep.py \
  --config configs/tinystories_60m_overnight.yaml \
  --checkpoint checkpoints/tinystories_60m_overnight/ckpt.pt \
  --vectors benchmarks/results/emotion_vectors_60m.pt \
  --output "$OUTPUT" \
  --emotions "$EMOTIONS" \
  --layers "$LAYERS" \
  --alphas "$ALPHAS" \
  --positions "$POSITIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS"

python -u scripts/08_score_steering_sweep.py --input "$OUTPUT"
