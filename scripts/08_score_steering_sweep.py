#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from sys import stderr
from typing import Any

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
    "playful": [
        "playful",
        "joke",
        "joked",
        "joking",
        "kidding",
        "pretend",
        "pretended",
        "silly",
        "funny",
        "giggle",
        "giggled",
        "laughed",
        "smile",
        "smiled",
    ],
    "serious": [
        "serious",
        "careful",
        "danger",
        "dangerous",
        "unsafe",
        "important",
        "problem",
        "hurt",
        "worried",
        "worry",
        "afraid",
        "scared",
        "safe",
        "help",
    ],
}

WORD_RE = re.compile(r"[a-z']+")
TOKEN_APPROX_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
STORY_BOUNDARY_PATTERNS = [
    re.compile(r"\bonce upon a time\b", re.IGNORECASE),
    re.compile(r"\bonce there (?:was|were|lived)\b", re.IGNORECASE),
    re.compile(r"\bthe end\b", re.IGNORECASE),
    re.compile(r"(?m)^\s*(?:story|title)\s*:", re.IGNORECASE),
]

SUMMARY_FIELDS = [
    "emotion",
    "layer",
    "alpha",
    "position",
    "n",
    "avg_delta",
    "avg_baseline_hits",
    "avg_steered_hits",
    "avg_elapsed_s",
    "avg_non_target_delta",
    "avg_baseline_non_target_hits",
    "avg_steered_non_target_hits",
    "avg_char_delta",
    "avg_baseline_chars",
    "avg_steered_chars",
    "avg_token_approx_delta",
    "avg_baseline_tokens_approx",
    "avg_steered_tokens_approx",
    "avg_story_boundary_delta",
    "avg_baseline_story_boundaries",
    "avg_steered_story_boundaries",
    "avg_repeat_3gram_delta",
    "avg_baseline_repeat_3gram_rate",
    "avg_steered_repeat_3gram_rate",
]


@dataclass
class WarningLimiter:
    max_warnings: int = 20
    emitted: int = 0
    suppressed: int = 0

    def warn(self, message: str) -> None:
        if self.max_warnings < 0 or self.emitted < self.max_warnings:
            print(f"warning: {message}", file=stderr)
            self.emitted += 1
            return
        self.suppressed += 1
        if self.suppressed == 1:
            print("warning: further warnings suppressed", file=stderr)


@dataclass
class JsonlParseStats:
    records: int = 0
    malformed_fragments: int = 0
    skipped_chars: int = 0


@dataclass
class RecordStats:
    skipped_records: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score steering sweep JSONL with simple lexicons.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed JSON fragments or invalid records instead of warning and skipping.",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=20,
        help="Maximum warnings to print before suppressing repeats. Use -1 for unlimited.",
    )
    return parser.parse_args()


def keyword_count(text: str, emotion: str) -> int:
    words = WORD_RE.findall(text.lower())
    word_text = " ".join(words)
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", word_text)) for term in LEXICON[emotion])


def non_target_keyword_count(text: str, target_emotion: str) -> int:
    return sum(keyword_count(text, emotion) for emotion in LEXICON if emotion != target_emotion)


def story_boundary_count(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in STORY_BOUNDARY_PATTERNS)


def token_approx_count(text: str) -> int:
    return len(TOKEN_APPROX_RE.findall(text))


def repetition_ngram_rate(text: str, n: int = 3) -> float:
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return 0.0
    total = len(words) - n + 1
    ngrams = [tuple(words[idx : idx + n]) for idx in range(total)]
    return 1.0 - (len(set(ngrams)) / total)


def text_metrics(text: str, target_emotion: str) -> dict[str, float]:
    return {
        "target_hits": float(keyword_count(text, target_emotion)),
        "non_target_hits": float(non_target_keyword_count(text, target_emotion)),
        "chars": float(len(text)),
        "tokens_approx": float(token_approx_count(text)),
        "story_boundaries": float(story_boundary_count(text)),
        "repeat_3gram_rate": repetition_ngram_rate(text),
    }


def _coerce_text(record: dict[str, Any], key: str) -> str:
    value = record[key]
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    emotion = str(record["emotion"])
    if emotion not in LEXICON:
        raise ValueError(f"unknown emotion {emotion!r}")

    baseline_text = _coerce_text(record, "baseline_text")
    steered_text = _coerce_text(record, "steered_text")
    baseline = text_metrics(baseline_text, emotion)
    steered = text_metrics(steered_text, emotion)

    return {
        "emotion": emotion,
        "layer": int(record["layer"]),
        "alpha": float(record["alpha"]),
        "position": str(record["position"]),
        "baseline_hits": baseline["target_hits"],
        "steered_hits": steered["target_hits"],
        "delta": steered["target_hits"] - baseline["target_hits"],
        "baseline_non_target_hits": baseline["non_target_hits"],
        "steered_non_target_hits": steered["non_target_hits"],
        "non_target_delta": steered["non_target_hits"] - baseline["non_target_hits"],
        "baseline_chars": baseline["chars"],
        "steered_chars": steered["chars"],
        "char_delta": steered["chars"] - baseline["chars"],
        "baseline_tokens_approx": baseline["tokens_approx"],
        "steered_tokens_approx": steered["tokens_approx"],
        "token_approx_delta": steered["tokens_approx"] - baseline["tokens_approx"],
        "baseline_story_boundaries": baseline["story_boundaries"],
        "steered_story_boundaries": steered["story_boundaries"],
        "story_boundary_delta": steered["story_boundaries"] - baseline["story_boundaries"],
        "baseline_repeat_3gram_rate": baseline["repeat_3gram_rate"],
        "steered_repeat_3gram_rate": steered["repeat_3gram_rate"],
        "repeat_3gram_delta": steered["repeat_3gram_rate"] - baseline["repeat_3gram_rate"],
        "elapsed_s": float(record.get("elapsed_s", 0.0) or 0.0),
    }


def _json_decode_error(
    path: Path,
    line_num: int,
    exc: json.JSONDecodeError,
) -> json.JSONDecodeError:
    return json.JSONDecodeError(
        f"{exc.msg} while parsing {path} line {line_num}",
        exc.doc,
        exc.pos,
    )


def _find_next_decodable_object(
    text: str,
    start_idx: int,
    decoder: json.JSONDecoder,
) -> tuple[int, Any, int] | None:
    idx = start_idx
    while True:
        idx = text.find("{", idx)
        if idx == -1:
            return None
        try:
            record, end_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        return idx, record, end_idx


def _preview_fragment(text: str, limit: int = 80) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        return collapsed[: limit - 3] + "..."
    return collapsed


def iter_json_records(
    path: Path,
    *,
    strict: bool = False,
    stats: JsonlParseStats | None = None,
    warnings: WarningLimiter | None = None,
):
    if stats is None:
        stats = JsonlParseStats()
    if warnings is None:
        warnings = WarningLimiter()

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            text = line.strip()
            idx = 0
            while idx < len(text):
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                if idx >= len(text):
                    break
                try:
                    record, end_idx = decoder.raw_decode(text, idx)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise _json_decode_error(path, line_num, exc) from exc

                    next_record = _find_next_decodable_object(text, idx + 1, decoder)
                    if next_record is None:
                        fragment = text[idx:]
                        stats.malformed_fragments += 1
                        stats.skipped_chars += len(fragment)
                        warnings.warn(
                            f"skipping malformed JSON fragment at {path}:{line_num}: "
                            f"{exc.msg}; fragment={_preview_fragment(fragment)!r}"
                        )
                        break

                    next_idx, _, _ = next_record
                    fragment = text[idx:next_idx]
                    stats.malformed_fragments += 1
                    stats.skipped_chars += len(fragment)
                    warnings.warn(
                        f"skipping malformed JSON fragment at {path}:{line_num}: "
                        f"{exc.msg}; fragment={_preview_fragment(fragment)!r}"
                    )
                    idx = next_idx
                    continue
                stats.records += 1
                yield record
                idx = end_idx


def summarize_groups(
    groups: dict[tuple[str, int, float, str], list[dict[str, float]]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
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
                "avg_non_target_delta": mean(item["non_target_delta"] for item in values),
                "avg_baseline_non_target_hits": mean(
                    item["baseline_non_target_hits"] for item in values
                ),
                "avg_steered_non_target_hits": mean(
                    item["steered_non_target_hits"] for item in values
                ),
                "avg_char_delta": mean(item["char_delta"] for item in values),
                "avg_baseline_chars": mean(item["baseline_chars"] for item in values),
                "avg_steered_chars": mean(item["steered_chars"] for item in values),
                "avg_token_approx_delta": mean(item["token_approx_delta"] for item in values),
                "avg_baseline_tokens_approx": mean(
                    item["baseline_tokens_approx"] for item in values
                ),
                "avg_steered_tokens_approx": mean(item["steered_tokens_approx"] for item in values),
                "avg_story_boundary_delta": mean(item["story_boundary_delta"] for item in values),
                "avg_baseline_story_boundaries": mean(
                    item["baseline_story_boundaries"] for item in values
                ),
                "avg_steered_story_boundaries": mean(
                    item["steered_story_boundaries"] for item in values
                ),
                "avg_repeat_3gram_delta": mean(item["repeat_3gram_delta"] for item in values),
                "avg_baseline_repeat_3gram_rate": mean(
                    item["baseline_repeat_3gram_rate"] for item in values
                ),
                "avg_steered_repeat_3gram_rate": mean(
                    item["steered_repeat_3gram_rate"] for item in values
                ),
            }
        )
    rows.sort(key=lambda row: (row["emotion"], -row["avg_delta"], row["layer"], row["alpha"]))
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else input_path.with_name(input_path.stem + "_summary.csv")
    )
    groups: dict[tuple[str, int, float, str], list[dict[str, float]]] = defaultdict(list)
    parse_stats = JsonlParseStats()
    record_stats = RecordStats()
    warnings = WarningLimiter(max_warnings=args.max_warnings)

    for record in iter_json_records(
        input_path,
        strict=args.strict,
        stats=parse_stats,
        warnings=warnings,
    ):
        try:
            scored = score_record(record)
        except (KeyError, TypeError, ValueError) as exc:
            if args.strict:
                raise
            record_stats.skipped_records += 1
            warnings.warn(f"skipping invalid record from {input_path}: {exc}")
            continue

        groups[
            (
                scored["emotion"],
                scored["layer"],
                scored["alpha"],
                scored["position"],
            )
        ].append(scored)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize_groups(groups)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_path}")
    if parse_stats.malformed_fragments or record_stats.skipped_records:
        print(
            "parse summary: "
            f"records={parse_stats.records} "
            f"malformed_fragments={parse_stats.malformed_fragments} "
            f"skipped_chars={parse_stats.skipped_chars} "
            f"invalid_records={record_stats.skipped_records}",
            file=stderr,
        )
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
