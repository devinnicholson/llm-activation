#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=generate_sweep_%j.out
#SBATCH --error=generate_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

MODELS="${MODELS:-8m,24m,60m}"
OUTPUT="${OUTPUT:-benchmarks/results/generation_sweep_${SLURM_JOB_ID}.jsonl}"

python -u scripts/04_generate_sweep.py \
  --models "$MODELS" \
  --output "$OUTPUT"
