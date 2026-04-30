from __future__ import annotations

import pytest

from scratch_llm.tokenizer import (
    SPECIAL_TOKENS,
    ScratchTokenizer,
    native_backend,
    save_bpe,
    train_bpe,
    train_bpe_to_file,
)

pytestmark = pytest.mark.skipif(
    native_backend() is None,
    reason="scratch_llm_native is not installed",
)


def test_rust_tokenizer_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    text = "High performance computing needs memory locality.\nUnicode works: héllò.\n"
    corpus.write_text(text, encoding="utf-8")

    vocab_size, merge_count, backend = train_bpe_to_file(
        input_path=corpus,
        vocab_size=320,
        output_path=tokenizer_path,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert backend == "rust"
    assert vocab_size > 256
    assert merge_count > 0
    assert tokenizer.decode(ids) == text


def test_rust_backend_reads_python_tokenizer_json(tmp_path):
    corpus = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "python_tokenizer.json"
    text = "FlashAttention is IO-aware exact attention.\n"
    corpus.write_text(text, encoding="utf-8")

    vocab, merges = train_bpe(
        input_path=corpus,
        vocab_size=300,
        special_tokens=SPECIAL_TOKENS,
    )
    save_bpe(
        vocab=vocab,
        merges=merges,
        path=tokenizer_path,
        special_tokens=SPECIAL_TOKENS,
    )

    rust_tokenizer = ScratchTokenizer.from_file(tokenizer_path, prefer_native=True)
    python_tokenizer = ScratchTokenizer.from_file(tokenizer_path, prefer_native=False)
    assert rust_tokenizer.encode(text, add_special_tokens=False) == python_tokenizer.encode(
        text,
        add_special_tokens=False,
    )
