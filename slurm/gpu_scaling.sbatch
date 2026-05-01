#!/bin/bash
#SBATCH --job-name=activation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SPACK_ENV_NAME=CS-2050-gpu
export REQUIRE_TORCH=1
source slurm/cluster_env.sh

CONFIG="${CONFIG:-configs/base_125m.yaml}"
RESUME="${RESUME:-0}"
mkdir -p benchmarks/results

echo "Running GPU scaling sweep"
echo "Config: $CONFIG"
echo "-------------------------"

for GPUS in 1 2 4; do
    echo "Running with $GPUS GPU(s)"
    TRAIN_ARGS=(--config "$CONFIG")
    if [ "$RESUME" = "1" ]; then
        TRAIN_ARGS+=(--resume)
    fi
    if [ "$GPUS" -eq 1 ]; then
        python scripts/02_train.py "${TRAIN_ARGS[@]}"
    else
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="$GPUS" \
            scripts/02_train.py \
            "${TRAIN_ARGS[@]}"
    fi
done
