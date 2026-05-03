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
    val_raw_files: list[str] | None = None,
    tokenizer_path: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    train_ratio: float,
) -> dict[str, int | str]:
    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    train_ids_list: list[int] = []

    for file_name in raw_files:
        text = Path(file_name).read_text(encoding="utf-8")
        train_ids_list.extend(tokenizer.encode(text, add_special_tokens=True))

    if len(train_ids_list) < 100:
        raise ValueError("Tokenized corpus is too small; add more text before training.")

    if val_raw_files is None:
        split_idx = int(len(train_ids_list) * train_ratio)
        split_idx = min(max(split_idx, 1), len(train_ids_list) - 1)
        val_ids_list = train_ids_list[split_idx:]
        train_ids_list = train_ids_list[:split_idx]
    else:
        val_ids_list: list[int] = []
        for file_name in val_raw_files:
            text = Path(file_name).read_text(encoding="utf-8")
            val_ids_list.extend(tokenizer.encode(text, add_special_tokens=True))
        if len(val_ids_list) < 100:
            raise ValueError("Tokenized validation corpus is too small.")

    dtype = dtype_for_vocab(tokenizer.vocab_size)
    train_ids = np.asarray(train_ids_list, dtype=dtype)
    val_ids = np.asarray(val_ids_list, dtype=dtype)

    train_output = Path(train_output)
    val_output = Path(val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(train_output, train_ids)
    np.save(val_output, val_ids)

    return {
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(dtype),
        "total_tokens": int(train_ids.shape[0] + val_ids.shape[0]),
        "train_tokens": int(train_ids.shape[0]),
        "val_tokens": int(val_ids.shape[0]),
    }


def load_token_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r")
