"""Atomic publication and CLI for final-run QC eligibility reports."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from nm_v4.aggregate_results import TABLE_COLUMNS
from nm_v4.checkpoints import sha256_file, sha256_json, utc_now
from nm_v4.qc_validation import (
    FLOW_COLUMNS,
    FORMAL_PERMUTATIONS,
    MODES,
    QCError,
    build_qc,
)
from nm_v4.runner_support import CANONICAL_DATASET_COUNT


QC_SEMANTIC_DEPENDENCIES = (
    "nm_v4/qc_report.py",
    "nm_v4/qc_validation.py",
    "nm_v4/qc_checkpoint_validation.py",
    "nm_v4/qc_contracts.py",
    "nm_v4/qc_nulls.py",
    "nm_v4/aggregate_results.py",
    "nm_v4/aggregation_schema.py",
    "nm_v4/aggregation_checkpoint_schema.py",
    "nm_v4/aggregation_uncertainty.py",
    "nm_v4/checkpoints.py",
    "nm_v4/configuration.py",
    "nm_v4/methods.py",
    "nm_v4/runner_support.py",
    "nm_v4/stability.py",
)
_QC_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QCError(f"{path}: expected a JSON object")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLOW_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(
            {column: row.get(column) for column in FLOW_COLUMNS} for row in rows
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _input_snapshot(run_dir: Path, table_dir: Path) -> dict[str, str]:
    files = {"run_manifest": run_dir / "manifest.json"}
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.json.gz"))
    if not checkpoints:
        raise QCError(f"{checkpoint_dir}: no checkpoint files")
    files.update({f"checkpoints/{path.name}": path for path in checkpoints})
    for name in TABLE_COLUMNS:
        candidates = [path for suffix in ("csv", "parquet")
                      if (path := table_dir / f"{name}.{suffix}").exists()]
        if len(candidates) != 1:
            raise QCError(f"{table_dir}: requires exactly one {name}.csv or .parquet")
        files[f"aggregation/{candidates[0].name}"] = candidates[0]
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise QCError("QC inputs must be regular, non-symlink files")
    return {label: sha256_file(path) for label, path in sorted(files.items())}


def _semantic_dependency_sha256(path: Path, relative_path: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QCError(
            f"QC semantic dependency is missing or unsafe: {relative_path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QCError(
                f"QC semantic dependency is missing or unsafe: {relative_path}"
            )
        digest = sha256()
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise QCError(
                f"QC semantic dependency changed while hashing: {relative_path}"
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _semantic_dependency_snapshot() -> dict[str, str]:
    return {
        relative_path: _semantic_dependency_sha256(
            _QC_SOURCE_ROOT / relative_path,
            relative_path,
        )
        for relative_path in QC_SEMANTIC_DEPENDENCIES
    }


def _publish(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".backup", dir=destination.parent
    )
    os.close(descriptor)
    backup = Path(name)
    backup.unlink()
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def run_qc(
    run_dir: Path | str,
    table_dir: Path | str,
    output_dir: Path | str,
    *,
    mode: str = "formal",
    overwrite: bool = False,
    expected_whitelist_count: int = CANONICAL_DATASET_COUNT,
) -> dict[str, Path]:
    """Validate and atomically publish the final eligibility-flow artifacts."""
    destination = Path(output_dir).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    run_path, table_path = Path(run_dir).resolve(), Path(table_dir).resolve()
    semantic_dependencies = _semantic_dependency_snapshot()
    before = _input_snapshot(run_path, table_path)
    flows, summary, _, audit = build_qc(
        run_path,
        table_path,
        mode=mode,
        expected_whitelist_count=expected_whitelist_count,
    )
    if before != _input_snapshot(run_path, table_path):
        raise QCError("runner/checkpoint/aggregation inputs changed during QC")
    if semantic_dependencies != _semantic_dependency_snapshot():
        raise QCError("QC semantic dependencies changed during QC")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        flow_path = staging / "dataset_eligibility_flow.csv"
        summary_path = staging / "qc_summary.json"
        _write_csv(flow_path, flows)
        _write_json(summary_path, summary)
        input_files = {key: value for key, value in before.items()
                       if not key.startswith("checkpoints/")}
        checkpoint_files = {key: value for key, value in before.items()
                            if key.startswith("checkpoints/")}
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "mode": mode,
            "qc_code_file_sha256": dict(semantic_dependencies),
            "qc_code_sha256": sha256_json(semantic_dependencies),
            "selected_ids_sha256": audit["selected_ids_sha256"],
            "checkpoint_file_set_sha256": sha256_json(checkpoint_files),
            "statistic_rule": summary["statistic_rule"],
            "permutation_inference_audit": summary[
                "permutation_inference_audit"
            ],
            "input_files": dict(sorted(input_files.items())),
            "output_files": {
                flow_path.name: sha256_file(flow_path),
                summary_path.name: sha256_file(summary_path),
            },
            "qc_status": summary["qc_status"],
            "formal_claims_allowed": summary["formal_claims_allowed"],
        }
        _write_json(staging / "qc_manifest.json", manifest)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if semantic_dependencies != _semantic_dependency_snapshot():
            raise QCError("QC semantic dependencies changed during QC")
        _publish(staging, destination, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "dataset_eligibility_flow": destination / "dataset_eligibility_flow.csv",
        "qc_summary": destination / "qc_summary.json",
        "qc_manifest": destination / "qc_manifest.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate final nm_v4 checkpoints and eight aggregation tables."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("table_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=MODES, default="formal")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run_qc(
            args.run_dir,
            args.table_dir,
            args.output_dir,
            mode=args.mode,
            overwrite=args.overwrite,
        )
    except (QCError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    summary = _read_json(outputs["qc_summary"])
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return int(summary["qc_status"] == "fail")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FLOW_COLUMNS",
    "FORMAL_PERMUTATIONS",
    "MODES",
    "QCError",
    "QC_SEMANTIC_DEPENDENCIES",
    "build_parser",
    "build_qc",
    "main",
    "run_qc",
]
