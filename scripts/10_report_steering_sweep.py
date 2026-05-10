#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from sys import stderr
from typing import Any


def _load_scorer_module():
    scorer_path = Path(__file__).with_name("08_score_steering_sweep.py")
    spec = importlib.util.spec_from_file_location("score_steering_sweep", scorer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _load_scorer_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Markdown report for steering sweep results."
    )
    parser.add_argument("--input", required=True, help="Steering sweep JSONL file.")
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary CSV from 08_score_steering_sweep.py. Defaults next to input.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown report path. Defaults next to input.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples-per-emotion", type=int, default=1)
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
    parser.add_argument(
        "--max-sample-chars",
        type=int,
        default=700,
        help="Maximum characters per baseline/steered sample in the report.",
    )
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _config_key(row: dict[str, Any]) -> tuple[str, int, float, str]:
    return (
        str(row["emotion"]),
        int(row["layer"]),
        float(row["alpha"]),
        str(row["position"]),
    )


def _md_escape(text: str, max_chars: int | None = None) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text.replace("|", "\\|").replace("\n", "<br>")


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _record_row(record: dict[str, Any]) -> str:
    prompt_idx = record.get("prompt_idx", "")
    return (
        f"| {record['emotion']} | {record['layer']} | {_fmt(record['alpha'])} | "
        f"{record['position']} | {prompt_idx} | {_fmt(record.get('quality_score', 0.0))} | "
        f"{_fmt(record['delta'])} | "
        f"{_fmt(record['non_target_delta'])} | {_fmt(record['token_approx_delta'])} | "
        f"{_fmt(record['repeat_3gram_delta'], 3)} |"
    )


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_records(
    input_path: Path,
    *,
    strict: bool,
    max_warnings: int,
) -> tuple[list[dict[str, Any]], Any, int]:
    parse_stats = SCORER.JsonlParseStats()
    warnings = SCORER.WarningLimiter(max_warnings=max_warnings)
    records: list[dict[str, Any]] = []
    skipped_records = 0

    for raw_record in SCORER.iter_json_records(
        input_path,
        strict=strict,
        stats=parse_stats,
        warnings=warnings,
    ):
        try:
            scored = SCORER.score_record(raw_record)
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise
            skipped_records += 1
            warnings.warn(f"skipping invalid record from {input_path}: {exc}")
            continue

        scored.update(
            {
                "prompt_idx": raw_record.get("prompt_idx", ""),
                "prompt": raw_record.get("prompt", ""),
                "baseline_text": raw_record.get("baseline_text", ""),
                "steered_text": raw_record.get("steered_text", ""),
            }
        )
        records.append(scored)

    return records, parse_stats, skipped_records


def top_configs_by_emotion(
    summary_rows: list[dict[str, str]],
    top_k: int,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        if not row.get("emotion"):
            continue
        grouped[row["emotion"]].append(row)

    for emotion, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                -_float(row, "avg_quality_score", _float(row, "avg_delta")),
                -_float(row, "avg_delta"),
                int(_float(row, "layer")),
                _float(row, "alpha"),
                row.get("position", ""),
            )
        )
        grouped[emotion] = rows[:top_k]
    return dict(sorted(grouped.items()))


def representative_records(
    records: list[dict[str, Any]],
    top_configs: dict[str, list[dict[str, str]]],
    samples_per_emotion: int,
) -> list[dict[str, Any]]:
    by_config: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_config[_config_key(record)].append(record)

    representatives: list[dict[str, Any]] = []
    for emotion in sorted(top_configs):
        selected = 0
        for config in top_configs[emotion]:
            key = (
                config["emotion"],
                _int(config, "layer"),
                _float(config, "alpha"),
                config.get("position", ""),
            )
            candidates = by_config.get(key, [])
            candidates.sort(
                key=lambda record: (
                    -float(record.get("quality_score", record["delta"])),
                    -float(record["delta"]),
                    str(record.get("prompt_idx", "")),
                    str(record.get("prompt", "")),
                )
            )
            for record in candidates:
                representatives.append(record)
                selected += 1
                if selected >= samples_per_emotion:
                    break
            if selected >= samples_per_emotion:
                break
    return representatives


def build_report(
    input_path: Path,
    summary_path: Path,
    summary_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    parse_stats: Any,
    skipped_records: int,
    *,
    top_k: int,
    samples_per_emotion: int,
    max_sample_chars: int,
) -> str:
    top_configs = top_configs_by_emotion(summary_rows, top_k=top_k)
    representatives = representative_records(
        records,
        top_configs,
        samples_per_emotion=samples_per_emotion,
    )

    lines = [
        "# Steering Sweep Report",
        "",
        f"- JSONL: `{input_path}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Valid records: {len(records)}",
        (
            f"- Skipped malformed fragments: {parse_stats.malformed_fragments} "
            f"({parse_stats.skipped_chars} chars)"
        ),
        f"- Skipped invalid records: {skipped_records}",
        "",
        "## Top Configs Per Emotion",
        "",
        (
            "| Emotion | Rank | Layer | Alpha | Position | n | Avg target delta | "
            "Avg quality | Avg steered hits | Avg non-target delta | Avg token delta | "
            "Avg repeat delta |"
        ),
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for emotion, rows in top_configs.items():
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {emotion} | {rank} | {_int(row, 'layer')} | "
                f"{_fmt(_float(row, 'alpha'))} | {row.get('position', '')} | "
                f"{_int(row, 'n')} | {_fmt(_float(row, 'avg_delta'))} | "
                f"{_fmt(_float(row, 'avg_quality_score', _float(row, 'avg_delta')))} | "
                f"{_fmt(_float(row, 'avg_steered_hits'))} | "
                f"{_fmt(_float(row, 'avg_non_target_delta'))} | "
                f"{_fmt(_float(row, 'avg_token_approx_delta'))} | "
                f"{_fmt(_float(row, 'avg_repeat_3gram_delta'), 3)} |"
            )

    if not top_configs:
        lines.append("| _No summary rows_ |  |  |  |  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Representative Records",
            "",
            (
                "| Emotion | Layer | Alpha | Position | Prompt idx | Quality | "
                "Target delta | Non-target delta | Token delta | Repeat delta |"
            ),
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if representatives:
        lines.extend(_record_row(record) for record in representatives)
    else:
        lines.append("| _No representative records_ |  |  |  |  |  |  |  |  |  |")

    lines.extend(["", "## Side-By-Side Samples", ""])
    if not representatives:
        lines.append("_No samples available._")
        return "\n".join(lines) + "\n"

    for record in representatives:
        prompt = str(record.get("prompt", "")).strip()
        lines.extend(
            [
                (
                    f"### {record['emotion']} "
                    f"layer={record['layer']} alpha={_fmt(record['alpha'])} "
                    f"position={record['position']} prompt={record.get('prompt_idx', '')}"
                ),
                "",
            ]
        )
        if prompt:
            lines.extend([f"Prompt: `{_md_escape(prompt, max_chars=180)}`", ""])
        lines.extend(
            [
                (
                    f"Target delta: {_fmt(record['delta'])}; "
                    f"quality: {_fmt(record.get('quality_score', 0.0))}; "
                    f"non-target delta: {_fmt(record['non_target_delta'])}; "
                    f"token delta: {_fmt(record['token_approx_delta'])}; "
                    f"repeat delta: {_fmt(record['repeat_3gram_delta'], 3)}"
                ),
                "",
                "| Baseline | Steered |",
                "|---|---|",
                (
                    f"| {_md_escape(str(record.get('baseline_text', '')), max_sample_chars)} "
                    f"| {_md_escape(str(record.get('steered_text', '')), max_sample_chars)} |"
                ),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    summary_path = (
        Path(args.summary)
        if args.summary
        else input_path.with_name(input_path.stem + "_summary.csv")
    )
    output_path = (
        Path(args.output) if args.output else input_path.with_name(input_path.stem + "_report.md")
    )

    summary_rows = read_summary(summary_path)
    records, parse_stats, skipped_records = load_records(
        input_path,
        strict=args.strict,
        max_warnings=args.max_warnings,
    )
    report = build_report(
        input_path,
        summary_path,
        summary_rows,
        records,
        parse_stats,
        skipped_records,
        top_k=args.top_k,
        samples_per_emotion=args.samples_per_emotion,
        max_sample_chars=args.max_sample_chars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote {output_path}")
    if parse_stats.malformed_fragments or skipped_records:
        print(
            "parse summary: "
            f"records={parse_stats.records} "
            f"malformed_fragments={parse_stats.malformed_fragments} "
            f"skipped_chars={parse_stats.skipped_chars} "
            f"invalid_records={skipped_records}",
            file=stderr,
        )


if __name__ == "__main__":
    main()
