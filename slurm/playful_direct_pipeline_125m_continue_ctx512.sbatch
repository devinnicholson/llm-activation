#!/bin/bash
#SBATCH --job-name=playful-125m-cont
#SBATCH --partition=gpu
#SBATCH --output=playful_direct_125m_continue_ctx512_%j.out
#SBATCH --error=playful_direct_125m_continue_ctx512_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export CONFIG="${CONFIG:-configs/tinystories_125m_full_ctx512_continue.yaml}"
export CHECKPOINT="${CHECKPOINT:-checkpoints/tinystories_125m_full_ctx512_continue/ckpt.pt}"
export PROMPT_BANK="${PROMPT_BANK:-prompt_banks/playful_vs_serious_direct.yaml}"
export VECTORS="${VECTORS:-benchmarks/results/playful_direct_vectors_125m_continue_ctx512.pt}"
export OUTPUT="${OUTPUT:-benchmarks/results/playful_direct_125m_continue_ctx512_${SLURM_JOB_ID}.jsonl}"
export LAYERS="${LAYERS:-3,4,5,6,7,8}"
export ALPHAS="${ALPHAS:-3,5,7.5,10,12}"
export POSITIONS="${POSITIONS:-last,all}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export TOP_K="${TOP_K:-100}"
export POOLING="${POOLING:-mean}"

bash slurm/playful_direct_pipeline_125m_ctx512.sbatch
