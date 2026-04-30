from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np

from scratch_llm.tokenizer import ScratchTokenizer


def resolve_raw_files(pattern: str) -> list[str]:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw text files matched pattern: {pattern}")
    return files


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def build_token_arrays(
    *,
    raw_files: list[str],
    tokenizer_path: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    train_ratio: float,
) -> dict[str, int | str]:
    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    ids: list[int] = []

    for file_name in raw_files:
        text = Path(file_name).read_text(encoding="utf-8")
        ids.extend(tokenizer.encode(text, add_special_tokens=True))

    if len(ids) < 100:
        raise ValueError("Tokenized corpus is too small; add more text before training.")

    split_idx = int(len(ids) * train_ratio)
    split_idx = min(max(split_idx, 1), len(ids) - 1)

    dtype = dtype_for_vocab(tokenizer.vocab_size)
    train_ids = np.asarray(ids[:split_idx], dtype=dtype)
    val_ids = np.asarray(ids[split_idx:], dtype=dtype)

    train_output = Path(train_output)
    val_output = Path(val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(train_output, train_ids)
    np.save(val_output, val_ids)

    return {
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(dtype),
        "total_tokens": len(ids),
        "train_tokens": int(train_ids.shape[0]),
        "val_tokens": int(val_ids.shape[0]),
    }


def load_token_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r")
