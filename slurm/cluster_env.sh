#!/bin/bash

# Shared cluster environment bootstrap based on the CS-2050 assignment scripts.
# Source this from sbatch scripts after `cd` into the project root.

set -euo pipefail

SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

if [ -f "$HOME/161588/spack/share/spack/setup-env.sh" ]; then
    . "$HOME/161588/spack/share/spack/setup-env.sh"
    spack env activate -p "$SPACK_ENV_NAME"
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Tasks: ${SLURM_NTASKS:-1}"
echo "Spack env: $SPACK_ENV_NAME"
echo "Python: $(command -v python || true)"
python --version || true
