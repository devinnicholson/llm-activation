#!/bin/bash
#SBATCH --job-name=scratch-llm-profile
#SBATCH --partition=gpu
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh
mkdir -p benchmarks/results

nsys profile \
  --output benchmarks/results/nsys_scratch_llm \
  --trace cuda,nvtx,osrt,cudnn,cublas \
  python scripts/02_train.py --config configs/base_125m.yaml
