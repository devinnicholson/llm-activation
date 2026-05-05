#!/bin/bash
#SBATCH --job-name=playful-threshold-fine
#SBATCH --partition=gpu
#SBATCH --output=playful_threshold_fine_%j.out
#SBATCH --error=playful_threshold_fine_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

OUTPUT="benchmarks/results/playful_threshold_fine_${SLURM_JOB_ID}.jsonl"

python -u scripts/07_steering_sweep.py \
  --config configs/tinystories_100m_full.yaml \
  --checkpoint checkpoints/tinystories_100m_full/best.pt \
  --vectors benchmarks/results/playful_serious_vectors_100m_full.pt \
  --output "$OUTPUT" \
  --emotions playful \
  --layers 9 \
  --alphas 0.055,0.06,0.065,0.07,0.075 \
  --positions last,all \
  --max-new-tokens 180 \
  --stop-at-eos

python -u scripts/08_score_steering_sweep.py --input "$OUTPUT"
python -u scripts/10_report_steering_sweep.py --input "$OUTPUT"
