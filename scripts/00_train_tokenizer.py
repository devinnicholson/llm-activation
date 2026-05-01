#!/usr/bin/env python
from __future__ import annotations

import argparse

from scratch_llm.config import load_config
from scratch_llm.data import resolve_raw_files
from scratch_llm.tokenizer import SPECIAL_TOKENS, train_bpe_to_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    files = resolve_raw_files(config["paths"]["raw_glob"])
    vocab_size, merge_count, backend = train_bpe_to_file(
        input_path=files,
        vocab_size=int(config["tokenizer"]["vocab_size"]),
        output_path=config["paths"]["tokenizer"],
        special_tokens=SPECIAL_TOKENS,
    )
    print(
        f"trained tokenizer on {len(files)} file(s): "
        f"{config['paths']['tokenizer']} vocab={vocab_size} merges={merge_count} "
        f"backend={backend}"
    )


if __name__ == "__main__":
    main()
