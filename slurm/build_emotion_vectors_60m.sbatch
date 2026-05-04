#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --output=emotion_vectors_60m_%j.out
#SBATCH --error=emotion_vectors_60m_%j.err
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
OUTPUT="${OUTPUT:-${VECTORS:-benchmarks/results/emotion_vectors_60m.pt}}"
LAYERS="${LAYERS:-all}"

python -u scripts/05_build_emotion_vectors.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --layers "$LAYERS"
