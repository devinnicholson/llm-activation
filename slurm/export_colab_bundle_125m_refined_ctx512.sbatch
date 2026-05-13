#!/bin/bash
#SBATCH --job-name=export-colab-125m-refined
#SBATCH --partition=gpu
#SBATCH --output=export_colab_bundle_125m_refined_ctx512_%j.out
#SBATCH --error=export_colab_bundle_125m_refined_ctx512_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/tinystories_125m_full_ctx512_continue.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_125m_full_ctx512_continue/ckpt.pt}"
VECTORS="${VECTORS:-benchmarks/results/playful_refined_vectors_125m_continue_ctx512.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-colab_bundle/playful_125m_continue_refined_ctx512}"
DTYPE="${DTYPE:-float16}"
EMOTION="${EMOTION:-playful}"
LAYER="${LAYER:-7}"
ALPHA="${ALPHA:-12}"
POSITION="${POSITION:-all}"
PROMPT="${PROMPT:-Once upon a time there was a little robot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-350}"

echo "Resolved Colab export settings:"
echo "  CONFIG=$CONFIG"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  VECTORS=$VECTORS"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  DTYPE=$DTYPE"
echo "  EMOTION=$EMOTION"
echo "  LAYER=$LAYER"
echo "  ALPHA=$ALPHA"
echo "  POSITION=$POSITION"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

python -u scripts/12_export_colab_bundle.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --vectors "$VECTORS" \
  --output-dir "$OUTPUT_DIR" \
  --dtype "$DTYPE" \
  --emotion "$EMOTION" \
  --layer "$LAYER" \
  --alpha "$ALPHA" \
  --position "$POSITION" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS"
