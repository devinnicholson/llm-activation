from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator
from glob import glob
from pathlib import Path

import numpy as np

from scratch_llm.tokenizer import ScratchTokenizer, TokenizerLike

_READ_CHUNK_CHARS = 1_048_576
_PROGRESS_MIN_BYTES = 64 * 1024 * 1024
_PROGRESS_INTERVAL_BYTES = 64 * 1024 * 1024


def resolve_raw_files(pattern: str) -> list[str]:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw text files matched pattern: {pattern}")
    return files


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def _token_array_typecode(dtype: np.dtype) -> str:
    if dtype == np.dtype(np.uint16):
        return "H"
    if dtype == np.dtype(np.uint32):
        return "I"
    raise ValueError(f"Unsupported token dtype: {dtype}")


def _array_to_numpy(tokens: array, dtype: np.dtype) -> np.ndarray:
    if tokens.itemsize == dtype.itemsize:
        return np.frombuffer(tokens, dtype=dtype)
    return np.asarray(tokens, dtype=dtype)


def _safe_cut_index(text: str, scan_start: int) -> int | None:
    start = max(scan_start, 1)
    last_cut = None
    for idx in range(start, len(text)):
        if text[idx].isspace() and not text[idx - 1].isspace():
            last_cut = idx
    return last_cut


def _iter_safe_text_segments(
    path: Path,
    *,
    read_chunk_chars: int = _READ_CHUNK_CHARS,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[str]:
    pending = ""

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(read_chunk_chars)
            if not chunk:
                break
            scan_start = len(pending)
            pending += chunk
            if on_progress is not None:
                on_progress(handle.tell())

            cut_idx = _safe_cut_index(pending, scan_start)
            if cut_idx is not None and cut_idx > 0:
                yield pending[:cut_idx]
                pending = pending[cut_idx:]

    if pending:
        yield pending


class _FileProgress:
    def __init__(
        self,
        *,
        label: str,
        path: Path,
        index: int,
        count: int,
        size_bytes: int,
        enabled: bool,
    ):
        self.label = label
        self.index = index
        self.count = count
        self.size_bytes = size_bytes
        self.enabled = enabled and size_bytes >= _PROGRESS_MIN_BYTES
        self.next_report = _PROGRESS_INTERVAL_BYTES
        if self.enabled:
            print(
                f"[data] Tokenizing {label} file {index}/{count}: "
                f"{path} ({_format_bytes(size_bytes)})",
                flush=True,
            )

    def update(self, position: int) -> None:
        if not self.enabled or self.size_bytes <= 0:
            return
        if position >= self.next_report and position < self.size_bytes:
            print(
                f"[data] {self.label} file {self.index}/{self.count}: "
                f"{_format_bytes(min(position, self.size_bytes))}/"
                f"{_format_bytes(self.size_bytes)} read",
                flush=True,
            )
            while self.next_report <= position:
                self.next_report += _PROGRESS_INTERVAL_BYTES

    def finish(self, token_count: int) -> None:
        if self.enabled:
            print(
                f"[data] Finished {self.label} file {self.index}/{self.count}: "
                f"{token_count:,} tokens",
                flush=True,
            )


def _tokenize_files(
    *,
    files: list[str],
    tokenizer: TokenizerLike,
    dtype: np.dtype,
    label: str,
    progress: bool,
) -> array:
    tokens = array(_token_array_typecode(dtype))
    paths = [Path(file_name) for file_name in files]
    total_bytes = sum(path.stat().st_size for path in paths)
    corpus_progress = progress and total_bytes >= _PROGRESS_MIN_BYTES
    if corpus_progress:
        print(
            f"[data] Tokenizing {label} corpus: {len(paths)} file(s), {_format_bytes(total_bytes)}",
            flush=True,
        )

    for index, path in enumerate(paths, start=1):
        before_file_tokens = len(tokens)
        file_progress = _FileProgress(
            label=label,
            path=path,
            index=index,
            count=len(paths),
            size_bytes=path.stat().st_size,
            enabled=progress,
        )
        for segment in _iter_safe_text_segments(path, on_progress=file_progress.update):
            tokens.extend(tokenizer.encode(segment, add_special_tokens=False))
        tokens.extend(tokenizer.encode("", add_special_tokens=True))
        file_progress.finish(len(tokens) - before_file_tokens)

    if corpus_progress:
        print(
            f"[data] Finished {label} corpus: {len(tokens):,} tokens",
            flush=True,
        )
    return tokens


def build_token_arrays(
    *,
    raw_files: list[str],
    val_raw_files: list[str] | None = None,
    tokenizer_path: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    train_ratio: float,
    progress: bool = False,
) -> dict[str, int | str]:
    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    dtype = dtype_for_vocab(tokenizer.vocab_size)
    train_tokens = _tokenize_files(
        files=raw_files,
        tokenizer=tokenizer,
        dtype=dtype,
        label="train",
        progress=progress,
    )

    if len(train_tokens) < 100:
        raise ValueError("Tokenized corpus is too small; add more text before training.")

    if val_raw_files is None:
        split_idx = int(len(train_tokens) * train_ratio)
        split_idx = min(max(split_idx, 1), len(train_tokens) - 1)
        all_ids = _array_to_numpy(train_tokens, dtype)
        train_ids = all_ids[:split_idx]
        val_ids = all_ids[split_idx:]
    else:
        val_tokens = _tokenize_files(
            files=val_raw_files,
            tokenizer=tokenizer,
            dtype=dtype,
            label="validation",
            progress=progress,
        )
        if len(val_tokens) < 100:
            raise ValueError("Tokenized validation corpus is too small.")
        train_ids = _array_to_numpy(train_tokens, dtype)
        val_ids = _array_to_numpy(val_tokens, dtype)

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
