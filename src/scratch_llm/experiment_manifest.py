from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def manifest_path_for_output(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.name}.manifest.json")


def ensure_outputs_available(paths: Iterable[str | Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output path(s): {formatted}. "
            "Pass --overwrite to replace them."
        )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def slurm_job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


class AtomicTextFile:
    def __init__(self, path: str | Path, *, overwrite: bool, encoding: str = "utf-8"):
        self.path = Path(path)
        self.overwrite = overwrite
        self.encoding = encoding
        self._handle: TextIO | None = None
        self._tmp_path: Path | None = None

    def __enter__(self) -> TextIO:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ensure_outputs_available([self.path], overwrite=self.overwrite)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        self._tmp_path = Path(tmp_name)
        self._handle = os.fdopen(fd, "w", encoding=self.encoding)
        return self._handle

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        tmp_path = self._tmp_path
        try:
            if self._handle is not None and not self._handle.closed:
                if exc_type is None:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
                self._handle.close()
            if exc_type is None and tmp_path is not None:
                commit_staged_file(tmp_path, self.path, overwrite=self.overwrite)
                self._tmp_path = None
        finally:
            if self._tmp_path is not None:
                try:
                    self._tmp_path.unlink()
                except FileNotFoundError:
                    pass
                self._tmp_path = None
        return False


def commit_staged_file(tmp_path: str | Path, final_path: str | Path, *, overwrite: bool) -> None:
    tmp = Path(tmp_path)
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        os.replace(tmp, final)
    else:
        try:
            os.link(tmp, final)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing output path: {final}. "
                "Pass --overwrite to replace it."
            ) from exc
        tmp.unlink()
    _fsync_parent(final.parent)


def write_json_atomic(payload: dict[str, Any], path: str | Path, *, overwrite: bool) -> None:
    with AtomicTextFile(path, overwrite=overwrite) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _fsync_parent(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
