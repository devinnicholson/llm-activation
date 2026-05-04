#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=steer_100m_full_%j.out
#SBATCH --error=steer_100m_full_%j.err
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

EMOTION="${EMOTION:-scared}"
LAYER="${LAYER:-7}"
ALPHA="${ALPHA:-2.0}"
POSITION="${POSITION:-all}"
PROMPT="${PROMPT:-Once upon a time there was a little robot}"

python -u scripts/06_steer_generation.py \
  --config configs/tinystories_100m_full.yaml \
  --checkpoint checkpoints/tinystories_100m_full/best.pt \
  --vectors benchmarks/results/emotion_vectors_100m_full.pt \
  --emotion "$EMOTION" \
  --layer "$LAYER" \
  --alpha "$ALPHA" \
  --position "$POSITION" \
  --prompt "$PROMPT"
