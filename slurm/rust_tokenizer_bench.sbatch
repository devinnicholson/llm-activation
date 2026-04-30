#!/bin/bash
#SBATCH --job-name=rust_bpe_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=rust_bpe_%j.out
#SBATCH --error=rust_bpe_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/cluster_env.sh

python -c 'import scratch_llm_native; print("native backend:", scratch_llm_native.backend_name())'
python scripts/00_train_tokenizer.py --config configs/tiny.yaml
