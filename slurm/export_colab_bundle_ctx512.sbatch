#!/bin/bash
#SBATCH --job-name=export-colab-ctx512
#SBATCH --partition=gpu
#SBATCH --output=export_colab_bundle_ctx512_%j.out
#SBATCH --error=export_colab_bundle_ctx512_%j.err
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

python -u scripts/12_export_colab_bundle.py \
  --config configs/tinystories_100m_full_ctx512.yaml \
  --checkpoint checkpoints/tinystories_100m_full_ctx512/best.pt \
  --vectors benchmarks/results/playful_serious_vectors_100m_full_ctx512.pt \
  --output-dir colab_bundle/playful_ctx512 \
  --dtype float16 \
  --emotion playful \
  --layer 4 \
  --alpha 1.5 \
  --position all \
  --prompt "Once upon a time there was a little robot" \
  --max-new-tokens 350
