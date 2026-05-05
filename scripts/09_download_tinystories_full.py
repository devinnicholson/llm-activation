#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

FILES = {
    "TinyStoriesV2-GPT4-train.txt": (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStoriesV2-GPT4-train.txt?download=true"
    ),
    "TinyStoriesV2-GPT4-valid.txt": (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStoriesV2-GPT4-valid.txt?download=true"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the full TinyStories GPT-4 corpus.")
    parser.add_argument("--out-dir", default="data/raw/tinystories_full")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def download(url: str, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"exists {output_path} size={output_path.stat().st_size:,}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-activation/1.0"})
    print(f"downloading {url}")
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=16 * 1024 * 1024)
    tmp_path.replace(output_path)
    print(f"wrote {output_path} size={output_path.stat().st_size:,}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    for name, url in FILES.items():
        download(url, out_dir / name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
