#!/usr/bin/env python
from __future__ import annotations

import argparse

from scratch_llm.config import load_config, save_yaml
from scratch_llm.data import build_token_arrays, resolve_raw_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize raw text into train/val arrays.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    files = resolve_raw_files(config["paths"]["raw_glob"])
    metadata = build_token_arrays(
        raw_files=files,
        tokenizer_path=config["paths"]["tokenizer"],
        train_output=config["paths"]["train_data"],
        val_output=config["paths"]["val_data"],
        train_ratio=float(config["data"]["train_ratio"]),
    )
    metadata["raw_files"] = files
    save_yaml(metadata, config["paths"]["metadata"])
    print(metadata)


if __name__ == "__main__":
    main()
