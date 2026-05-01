from __future__ import annotations

import torch

from scratch_llm.model import TransformerConfig, TransformerLM
from scratch_llm.tokenizer import ScratchTokenizer, save_bpe, train_bpe


def test_bpe_tokenizer_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    text = "High performance computing needs memory locality.\n"
    corpus.write_text(text, encoding="utf-8")

    vocab, merges = train_bpe(
        input_path=corpus,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    save_bpe(
        vocab=vocab,
        merges=merges,
        path=tokenizer_path,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids) == text


def test_bpe_trains_from_multiple_files(tmp_path):
    corpus_a = tmp_path / "a.txt"
    corpus_b = tmp_path / "b.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    text_a = "High performance computing needs memory locality.\n"
    text_b = "Distributed training needs careful communication.\n"
    corpus_a.write_text(text_a, encoding="utf-8")
    corpus_b.write_text(text_b, encoding="utf-8")

    vocab, merges = train_bpe(
        input_path=[corpus_a, corpus_b],
        vocab_size=320,
        special_tokens=["<|endoftext|>"],
    )
    save_bpe(
        vocab=vocab,
        merges=merges,
        path=tokenizer_path,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    assert tokenizer.decode(tokenizer.encode(text_a, add_special_tokens=False)) == text_a
    assert tokenizer.decode(tokenizer.encode(text_b, add_special_tokens=False)) == text_b


def test_transformer_forward_shape_and_loss():
    config = TransformerConfig(
        vocab_size=128,
        context_length=16,
        num_layers=2,
        num_heads=4,
        d_model=32,
        d_ff=96,
        rope_theta=10000.0,
        dropout=0.0,
    )
    model = TransformerLM(config)
    idx = torch.randint(0, config.vocab_size, (3, config.context_length))
    targets = torch.randint(0, config.vocab_size, (3, config.context_length))
    logits, loss = model(idx, targets)

    assert logits.shape == (3, config.context_length, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()


def test_transformer_forward_with_flash_attention():
    config = TransformerConfig(
        vocab_size=128,
        context_length=16,
        num_layers=2,
        num_heads=4,
        d_model=32,
        d_ff=96,
        rope_theta=10000.0,
        dropout=0.0,
        attention_impl="flash_pytorch",
    )
    model = TransformerLM(config)
    idx = torch.randint(0, config.vocab_size, (3, config.context_length))
    targets = torch.randint(0, config.vocab_size, (3, config.context_length))
    logits, loss = model(idx, targets)

    assert logits.shape == (3, config.context_length, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
