#!/bin/bash

set -euo pipefail

SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050-gpu}"

if [ -f "$HOME/161588/spack/share/spack/setup-env.sh" ]; then
    . "$HOME/161588/spack/share/spack/setup-env.sh"
    spack env activate -p "$SPACK_ENV_NAME"
fi

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

if [ -n "${TORCH_INDEX_URL:-}" ]; then
    python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -e ".[dev]"

if command -v maturin >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    if ! (
        cd native/scratch_llm_native
        maturin develop --release
    ); then
        echo "Rust native backend build failed; continuing with the Python tokenizer backend."
    fi
else
    echo "Skipping Rust native backend build because maturin or cargo is unavailable."
fi

python scripts/00_train_tokenizer.py --config configs/tiny.yaml
python scripts/01_prepare_dataset.py --config configs/tiny.yaml

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available now:", torch.cuda.is_available())
print("cluster venv setup complete")
PY
