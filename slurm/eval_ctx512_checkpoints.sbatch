#!/bin/bash
#SBATCH --job-name=eval-ctx512
#SBATCH --partition=gpu
#SBATCH --output=eval_ctx512_checkpoints_%j.out
#SBATCH --error=eval_ctx512_checkpoints_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

EVAL_ITERS="${EVAL_ITERS:-200}"

eval_one() {
  local config="$1"
  local checkpoint="$2"
  local label="$3"

  if [ ! -f "$checkpoint" ]; then
    echo "missing_${label}=$checkpoint"
    return
  fi

  echo
  echo "===== $label ====="
  python -u scripts/13_eval_checkpoint.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --eval-iters "$EVAL_ITERS"
}

eval_one \
  "configs/tinystories_100m_full_ctx512.yaml" \
  "checkpoints/tinystories_100m_full_ctx512/best.pt" \
  "current_107m_best"

eval_one \
  "configs/tinystories_100m_full_ctx512.yaml" \
  "checkpoints/tinystories_100m_full_ctx512/ckpt.pt" \
  "current_107m_final"

eval_one \
  "configs/tinystories_125m_full_ctx512.yaml" \
  "checkpoints/tinystories_125m_full_ctx512/best.pt" \
  "current_125m_best"

eval_one \
  "configs/tinystories_125m_full_ctx512.yaml" \
  "checkpoints/tinystories_125m_full_ctx512/ckpt.pt" \
  "current_125m_final"

eval_one \
  "configs/tinystories_125m_full_ctx512_continue.yaml" \
  "checkpoints/tinystories_125m_full_ctx512_continue/best.pt" \
  "continued_125m_best"

eval_one \
  "configs/tinystories_125m_full_ctx512_continue.yaml" \
  "checkpoints/tinystories_125m_full_ctx512_continue/ckpt.pt" \
  "continued_125m_final"

eval_one \
  "configs/tinystories_107m_full_ctx512_tuned.yaml" \
  "checkpoints/tinystories_107m_full_ctx512_tuned/best.pt" \
  "tuned_107m_best"

eval_one \
  "configs/tinystories_107m_full_ctx512_tuned.yaml" \
  "checkpoints/tinystories_107m_full_ctx512_tuned/ckpt.pt" \
  "tuned_107m_final"

eval_one \
  "configs/tinystories_125m_full_ctx512_continue_low_lr.yaml" \
  "checkpoints/tinystories_125m_full_ctx512_continue_low_lr/best.pt" \
  "low_lr_125m_best"

eval_one \
  "configs/tinystories_125m_full_ctx512_continue_low_lr.yaml" \
  "checkpoints/tinystories_125m_full_ctx512_continue_low_lr/ckpt.pt" \
  "low_lr_125m_final"
