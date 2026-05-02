#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

LEXICON = {
    "happy": [
        "happy",
        "joy",
        "glad",
        "excited",
        "smile",
        "smiled",
        "laugh",
        "laughed",
        "fun",
        "wonderful",
        "proud",
        "love",
        "loved",
    ],
    "sad": [
        "sad",
        "cry",
        "cried",
        "tears",
        "lonely",
        "missed",
        "upset",
        "unhappy",
        "sorry",
        "lost",
        "alone",
        "hurt",
    ],
    "scared": [
        "scared",
        "afraid",
        "fear",
        "frightened",
        "hid",
        "hide",
        "worried",
        "dark",
        "noise",
        "strange",
        "shake",
        "shook",
        "monster",
    ],
    "calm": [
        "calm",
        "peaceful",
        "quiet",
        "relaxed",
        "safe",
        "softly",
        "gentle",
        "rested",
        "sleep",
        "asleep",
        "warm",
        "content",
    ],
    "curious": [
        "curious",
        "wondered",
        "question",
        "asked",
        "explore",
        "explored",
        "discover",
        "discovered",
        "learn",
        "looked",
        "found",
        "strange",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score steering sweep JSONL with simple lexicons.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def keyword_count(text: str, emotion: str) -> int:
    words = re.findall(r"[a-z']+", text.lower())
    word_text = " ".join(words)
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", word_text)) for term in LEXICON[emotion])


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_summary.csv")
    )
    groups: dict[tuple[str, int, float, str], list[dict[str, float]]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            emotion = record["emotion"]
            baseline_hits = keyword_count(record["baseline_text"], emotion)
            steered_hits = keyword_count(record["steered_text"], emotion)
            groups[
                (
                    emotion,
                    int(record["layer"]),
                    float(record["alpha"]),
                    record["position"],
                )
            ].append(
                {
                    "baseline_hits": baseline_hits,
                    "steered_hits": steered_hits,
                    "delta": steered_hits - baseline_hits,
                    "elapsed_s": float(record["elapsed_s"]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (emotion, layer, alpha, position), values in groups.items():
        rows.append(
            {
                "emotion": emotion,
                "layer": layer,
                "alpha": alpha,
                "position": position,
                "n": len(values),
                "avg_delta": mean(item["delta"] for item in values),
                "avg_baseline_hits": mean(item["baseline_hits"] for item in values),
                "avg_steered_hits": mean(item["steered_hits"] for item in values),
                "avg_elapsed_s": mean(item["elapsed_s"] for item in values),
            }
        )

    rows.sort(key=lambda row: (row["emotion"], -row["avg_delta"], row["layer"], row["alpha"]))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "emotion",
                "layer",
                "alpha",
                "position",
                "n",
                "avg_delta",
                "avg_baseline_hits",
                "avg_steered_hits",
                "avg_elapsed_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_path}")
    for emotion in sorted(LEXICON):
        best = [row for row in rows if row["emotion"] == emotion][:3]
        print(f"\n{emotion}")
        for row in best:
            print(
                f"  layer={row['layer']} alpha={row['alpha']} position={row['position']} "
                f"avg_delta={row['avg_delta']:.2f}"
            )


if __name__ == "__main__":
    main()
