from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol

import regex as re

SPECIAL_TOKENS = ["<|endoftext|>"]
GPT2_PRETOKENIZER_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class TokenizerLike(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...


def native_backend():
    try:
        return import_module("scratch_llm_native")
    except ImportError:
        return None


def native_backend_name() -> str | None:
    native = native_backend()
    if native is None:
        return None
    return str(native.backend_name())


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[tuple[str, bool]]:
    if not special_tokens:
        return [(text, False)]
    escaped_tokens = [
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    ]
    pattern = "(" + "|".join(escaped_tokens) + ")"
    pieces = re.split(pattern, text)
    special_set = set(special_tokens)
    return [(piece, piece in special_set) for piece in pieces if piece]


def _pretokenize(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for piece, is_special in _split_on_special_tokens(text, special_tokens):
        if is_special:
            continue
        for match in GPT2_PRETOKENIZER_PATTERN.finditer(piece):
            token = match.group(0).encode("utf-8")
            counts[tuple(bytes([byte]) for byte in token)] += 1
    return counts


def _normalize_input_paths(input_path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(input_path, str | Path):
        return [Path(input_path)]
    return [Path(path) for path in input_path]


def _pair_counts(word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, count in word_counts.items():
        for left, right in zip(word, word[1:], strict=False):
            counts[(left, right)] += count
    return counts


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged = []
    idx = 0
    while idx < len(word):
        if idx < len(word) - 1 and word[idx] == pair[0] and word[idx + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            idx += 2
        else:
            merged.append(word[idx])
            idx += 1
    return tuple(merged)


def train_bpe(
    *,
    input_path: str | Path | Sequence[str | Path],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer without delegating to a tokenizer library."""
    special_tokens = special_tokens or SPECIAL_TOKENS
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit all byte tokens plus special tokens.")

    input_paths = _normalize_input_paths(input_path)
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for path in input_paths:
        word_counts.update(_pretokenize(path.read_text(encoding="utf-8"), special_tokens))

    vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pairs = _pair_counts(word_counts)
        if not pairs:
            break
        best_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        word_counts = Counter(
            {_merge_word(word, best_pair): count for word, count in word_counts.items()}
        )

    return vocab, merges


def save_bpe(
    *,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    path: str | Path,
    special_tokens: list[str] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "special_tokens": special_tokens or SPECIAL_TOKENS,
        "vocab": [[idx, token.hex()] for idx, token in sorted(vocab.items())],
        "merges": [[left.hex(), right.hex()] for left, right in merges],
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def load_bpe(path: str | Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {int(idx): bytes.fromhex(token_hex) for idx, token_hex in payload["vocab"]}
    merges = [
        (bytes.fromhex(left_hex), bytes.fromhex(right_hex))
        for left_hex, right_hex in payload["merges"]
    ]
    return vocab, merges, list(payload.get("special_tokens", SPECIAL_TOKENS))


def train_bpe_to_file(
    *,
    input_path: str | Path | Sequence[str | Path],
    vocab_size: int,
    output_path: str | Path,
    special_tokens: list[str] | None = None,
    prefer_native: bool = True,
) -> tuple[int, int, str]:
    special_tokens = special_tokens or SPECIAL_TOKENS
    input_paths = _normalize_input_paths(input_path)
    native = native_backend() if prefer_native else None
    if native is not None and len(input_paths) == 1:
        trained_vocab_size, merge_count = native.train_bpe_to_file(
            str(input_paths[0]),
            vocab_size,
            str(output_path),
            special_tokens,
        )
        return int(trained_vocab_size), int(merge_count), "rust"

    vocab, merges = train_bpe(
        input_path=input_paths,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    save_bpe(
        vocab=vocab,
        merges=merges,
        path=output_path,
        special_tokens=special_tokens,
    )
    return len(vocab), len(merges), "python"


class NativeScratchTokenizer:
    def __init__(self, path: str | Path):
        native = native_backend()
        if native is None:
            raise ImportError("scratch_llm_native is not installed.")
        self.path = Path(path)
        self._native = native
        self._vocab_size = int(native.tokenizer_vocab_size(str(self.path)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(self._native.encode_file(str(self.path), text, add_special_tokens))

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return str(self._native.decode_file(str(self.path), ids, skip_special_tokens))


class ScratchTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.token_to_id
        }
        self.special_token_ids = set(self.special_token_to_id.values())

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        prefer_native: bool = True,
    ) -> TokenizerLike:
        if prefer_native and native_backend() is not None:
            return NativeScratchTokenizer(path)
        vocab, merges, special_tokens = load_bpe(path)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        for piece, is_special in _split_on_special_tokens(text, self.special_tokens):
            if is_special:
                ids.append(self.special_token_to_id[piece])
                continue
            for match in GPT2_PRETOKENIZER_PATTERN.finditer(piece):
                ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))

        if add_special_tokens and "<|endoftext|>" in self.special_token_to_id:
            ids.append(self.special_token_to_id["<|endoftext|>"])
        return ids

    def _encode_pretoken(self, token: bytes) -> list[int]:
        pieces = tuple(bytes([byte]) for byte in token)
        while len(pieces) > 1:
            candidate_pairs = [
                pair for pair in zip(pieces, pieces[1:], strict=False) if pair in self.merge_ranks
            ]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda pair: self.merge_ranks[pair])
            pieces = _merge_word(pieces, best_pair)
        return [self.token_to_id[piece] for piece in pieces]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [idx for idx in ids if int(idx) not in self.special_token_ids]
        data = b"".join(self.vocab[int(idx)] for idx in ids)
        return data.decode("utf-8", errors="replace")
