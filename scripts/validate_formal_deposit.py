"""Validate a sanitized Scientific Data formal-deposit directory."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

try:
    from scripts.formal_deposit_contract import (
        DEFAULT_RECORDS,
        DEFAULT_SOURCE_CATALOG_CONTRACT,
        HUMAN_REDACT_FIELDS,
        SOURCE_CATALOG_FIELDS,
        RecordContract,
        forbidden_identifier_kind,
        forbidden_json_identifier,
        raw_matrix_key,
        sha256_file,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from formal_deposit_contract import (  # type: ignore[no-redef]
        DEFAULT_RECORDS,
        DEFAULT_SOURCE_CATALOG_CONTRACT,
        HUMAN_REDACT_FIELDS,
        SOURCE_CATALOG_FIELDS,
        RecordContract,
        forbidden_identifier_kind,
        forbidden_json_identifier,
        raw_matrix_key,
        sha256_file,
    )

CHECKSUM_PATTERN = re.compile(r"([0-9a-f]{64})  (.+)\Z")
CONTROL_FILES = {
    "DATA_LICENSE",
    "LICENSE",
    "README.md",
    "catalog/source_accession_and_license.csv",
    "release_manifest.json",
    "checksums.sha256",
    "schema/data_dictionary.csv",
}


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    table_count: int
    checkpoint_count: int
    validated_file_count: int

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class InventoryCheckpoint:
    path: str
    sha256: str
    size_bytes: int


def _repository_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value and not path.is_absolute() and ".." not in path.parts)


def _validate_layout(
    root: Path,
    files: Mapping[str, Path],
    contracts: Mapping[str, RecordContract],
    source_catalog_contract: RecordContract,
    expected_checkpoints: int,
    require_validation_report: bool,
) -> list[str]:
    errors: list[str] = []
    required = {
        *(f"tables/{name}" for name in contracts),
        *CONTROL_FILES,
    }
    if source_catalog_contract.relative_source != (
        "catalog/source_accession_and_license.csv"
    ):
        errors.append("source catalog contract path is invalid")
    if require_validation_report:
        required.add("VALIDATION_REPORT.md")
    missing = sorted(required - set(files))
    if missing:
        errors.append(f"missing required files: {missing}")

    checkpoints = {
        relative
        for relative in files
        if relative.startswith("checkpoints/")
    }
    if len(checkpoints) != expected_checkpoints:
        errors.append(
            f"expected {expected_checkpoints} checkpoints, found {len(checkpoints)}"
        )
    expected_files = required | checkpoints
    extra = sorted(set(files) - expected_files)
    if extra:
        errors.append(
            "unexpected files (raw matrices/manifests are forbidden): "
            f"{extra}"
        )
    for relative, path in files.items():
        if path.is_symlink():
            errors.append(f"symbolic links are forbidden: {relative}")
        if relative.startswith("checkpoints/") and not relative.endswith(
            ".json.gz"
        ):
            errors.append(f"unexpected checkpoint payload: {relative}")
    return errors


def _scan_text_file(path: Path, relative: str) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                finding = forbidden_identifier_kind(line)
                if finding is not None:
                    errors.append(
                        f"{relative}:{line_number}: forbidden identifier ({finding})"
                    )
                    break
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot scan {relative}: {exc}")
    return errors


def _validate_csv(
    path: Path,
    filename: str,
    contract: RecordContract,
) -> tuple[list[str], tuple[str, ...], list[dict[str, str]]]:
    errors: list[str] = []
    retained_rows: list[dict[str, str]] = []
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if len(fields) != contract.fields or len(set(fields)) != len(fields):
                errors.append(
                    f"{filename}: expected {contract.fields} unique fields, "
                    f"found {len(fields)}"
                )
            raw_fields = [
                field
                for field in fields
                if field.casefold()
                in {
                    "x",
                    "raw_x",
                    "matrix",
                    "raw_matrix",
                    "data_matrix",
                    "feature_matrix",
                    "expression_matrix",
                }
            ]
            if raw_fields:
                errors.append(f"{filename}: raw matrix fields are forbidden")
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                if None in row:
                    errors.append(f"{filename}:{row_number}: malformed CSV row")
                    continue
                for value in row.values():
                    finding = forbidden_identifier_kind(value)
                    if finding is not None:
                        errors.append(
                            f"{filename}:{row_number}: forbidden identifier "
                            f"({finding})"
                        )
                        break
                if filename == "failures.csv":
                    if row.get("traceback", "") != "":
                        errors.append(
                            f"{filename}:{row_number}: traceback was not cleared"
                        )
                    checkpoint_path = row.get("checkpoint_path", "")
                    if checkpoint_path and not checkpoint_path.startswith(
                        "checkpoints/"
                    ):
                        errors.append(
                            f"{filename}:{row_number}: checkpoint path is not relative"
                        )
                elif filename == "human_cohort_adjudication_v2_manual.csv":
                    uncleared = [
                        field
                        for field in HUMAN_REDACT_FIELDS
                        if row.get(field, "") != ""
                    ]
                    if uncleared:
                        errors.append(
                            f"{filename}:{row_number}: disclosure fields not cleared: "
                            f"{sorted(uncleared)}"
                        )
                if filename == "source_accession_and_license.csv":
                    parsed = urlparse(row.get("source_url", ""))
                    if (
                        parsed.scheme not in {"http", "https"}
                        or not parsed.netloc
                        or parsed.username is not None
                        or parsed.password is not None
                    ):
                        errors.append(
                            f"{filename}:{row_number}: unsafe source_url"
                        )
                if filename in {
                    "run_inventory.csv",
                    "source_accession_and_license.csv",
                }:
                    retained_rows.append(dict(row))
            if row_count != contract.rows:
                errors.append(
                    f"{filename}: expected {contract.rows} rows, found {row_count}"
                )
    except (csv.Error, OSError, UnicodeError) as exc:
        fields = ()
        errors.append(f"cannot validate {filename}: {exc}")
    return errors, fields, retained_rows


def _inventory_contract(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[str], dict[str, InventoryCheckpoint]]:
    errors: list[str] = []
    inventory: dict[str, InventoryCheckpoint] = {}
    for row_number, row in enumerate(rows, start=2):
        dataset_id = row.get("dataset_id", "").strip()
        relative = row.get("checkpoint_path", "").strip()
        digest = row.get("checkpoint_sha256", "").strip()
        size_text = row.get("checkpoint_size_bytes", "").strip()
        if not dataset_id or dataset_id in inventory:
            errors.append(f"run_inventory.csv:{row_number}: duplicate/empty dataset_id")
            continue
        expected_relative = f"checkpoints/{dataset_id}.json.gz"
        if relative != expected_relative:
            errors.append(
                f"run_inventory.csv:{row_number}: expected path {expected_relative}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(
                f"run_inventory.csv:{row_number}: invalid checkpoint hash"
            )
        try:
            size = int(size_text)
            if size < 1:
                raise ValueError
        except ValueError:
            errors.append(
                f"run_inventory.csv:{row_number}: invalid checkpoint size"
            )
            size = -1
        inventory[dataset_id] = InventoryCheckpoint(relative, digest, size)
    return errors, inventory


def _validate_checkpoint(
    root: Path,
    dataset_id: str,
    contract: InventoryCheckpoint,
) -> list[str]:
    errors: list[str] = []
    path = root / contract.path
    if not path.is_file():
        return [f"missing checkpoint for {dataset_id}: {contract.path}"]
    if path.stat().st_size != contract.size_bytes:
        errors.append(f"{contract.path}: inventory size mismatch")
    if sha256_file(path) != contract.sha256:
        errors.append(f"{contract.path}: inventory hash mismatch")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{contract.path}: invalid gzip JSON: {exc}"]
    if not isinstance(payload, dict):
        return [f"{contract.path}: checkpoint root is not an object"]
    if payload.get("dataset_id") != dataset_id:
        errors.append(f"{contract.path}: dataset_id mismatch")
    matrix_key = raw_matrix_key(payload)
    if matrix_key is not None:
        errors.append(f"{contract.path}: raw matrix key is forbidden: {matrix_key}")
    finding = forbidden_json_identifier(payload)
    if finding is not None:
        errors.append(f"{contract.path}: forbidden identifier ({finding})")
    checkpoint_errors = payload.get("errors", [])
    if not isinstance(checkpoint_errors, list) or any(
        not isinstance(item, dict) for item in checkpoint_errors
    ):
        errors.append(f"{contract.path}: errors must be a list of objects")
    elif any(item.get("traceback", "") != "" for item in checkpoint_errors):
        errors.append(f"{contract.path}: errors[].traceback was not cleared")
    return errors


def _validate_dictionary(
    path: Path,
    table_fields: Mapping[str, Sequence[str]],
) -> list[str]:
    errors: list[str] = []
    expected_header = (
        "file",
        "field",
        "type",
        "nullable",
        "scope",
        "description",
    )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_header:
                errors.append("data dictionary header does not match contract")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"cannot validate data dictionary: {exc}"]
    expected_pairs = {
        (filename, field)
        for filename, fields in table_fields.items()
        for field in fields
    }
    observed_pairs = {(row.get("file", ""), row.get("field", "")) for row in rows}
    if observed_pairs != expected_pairs:
        errors.append(
            "data dictionary field coverage does not match the release tables"
        )
    descriptions = {row.get("field", ""): row.get("description", "") for row in rows}
    for field in ("test_index", "group_code"):
        if "dataset-local integer" not in descriptions.get(field, ""):
            errors.append(
                f"data dictionary must document {field} as dataset-local integer"
            )
    return errors


def _validate_manifest(
    root: Path,
    path: Path,
    expected_payload_paths: set[str],
    table_count: int,
    checkpoint_count: int,
) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot validate release_manifest.json: {exc}"]
    finding = forbidden_json_identifier(payload)
    if finding is not None:
        errors.append(f"release_manifest.json: forbidden identifier ({finding})")
    if payload.get("table_count") != table_count:
        errors.append("release manifest table_count mismatch")
    if payload.get("checkpoint_count") != checkpoint_count:
        errors.append("release manifest checkpoint_count mismatch")
    entries = payload.get("payload_files")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        return [*errors, "release manifest payload_files must be a list"]
    entry_paths = [str(entry.get("path", "")) for entry in entries]
    if len(entry_paths) != len(set(entry_paths)):
        errors.append("release manifest has duplicate payload paths")
    if set(entry_paths) != expected_payload_paths:
        errors.append("release manifest payload path coverage mismatch")
    for entry in entries:
        relative = str(entry.get("path", ""))
        if not _safe_relative_path(relative):
            errors.append(f"release manifest has unsafe path: {relative}")
            continue
        artifact = root / relative
        if not artifact.is_file():
            continue
        if entry.get("size_bytes") != artifact.stat().st_size:
            errors.append(f"release manifest size mismatch: {relative}")
        if entry.get("sha256") != sha256_file(artifact):
            errors.append(f"release manifest hash mismatch: {relative}")
    return errors


def _validate_checksums(root: Path, files: Mapping[str, Path]) -> list[str]:
    errors: list[str] = []
    checksum_path = root / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"cannot validate checksums.sha256: {exc}"]
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            errors.append(f"checksums.sha256:{line_number}: malformed checksum")
            continue
        digest, relative = match.groups()
        if not _safe_relative_path(relative) or relative in entries:
            errors.append(f"checksums.sha256:{line_number}: unsafe/duplicate path")
            continue
        entries[relative] = digest
    expected = set(files) - {"checksums.sha256", "VALIDATION_REPORT.md"}
    if set(entries) != expected:
        errors.append("checksum path coverage mismatch")
    for relative, expected_digest in entries.items():
        artifact = files.get(relative)
        if artifact is not None and sha256_file(artifact) != expected_digest:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def validate_formal_deposit(
    root: Path,
    *,
    contracts: Mapping[str, RecordContract] = DEFAULT_RECORDS,
    source_catalog_contract: RecordContract = DEFAULT_SOURCE_CATALOG_CONTRACT,
    expected_checkpoints: int = 1_102,
    require_validation_report: bool = True,
) -> ValidationResult:
    """Return all structural, integrity and disclosure-control failures."""
    deposit_root = Path(root).resolve(strict=False)
    if not deposit_root.is_dir():
        return ValidationResult((f"not a directory: {deposit_root}",), 0, 0, 0)
    files = _repository_files(deposit_root)
    errors = _validate_layout(
        deposit_root,
        files,
        contracts,
        source_catalog_contract,
        expected_checkpoints,
        require_validation_report,
    )

    table_fields: dict[str, tuple[str, ...]] = {}
    inventory_rows: list[dict[str, str]] = []
    for filename, contract in contracts.items():
        path = deposit_root / "tables" / filename
        if not path.is_file():
            continue
        csv_errors, fields, retained_rows = _validate_csv(
            path, filename, contract
        )
        errors.extend(csv_errors)
        table_fields[filename] = fields
        if filename == "run_inventory.csv":
            inventory_rows = retained_rows
    canonical_table_count = len(table_fields)

    source_catalog_rows: list[dict[str, str]] = []
    source_catalog_path = deposit_root / source_catalog_contract.relative_source
    if source_catalog_path.is_file():
        catalog_errors, catalog_fields, source_catalog_rows = _validate_csv(
            source_catalog_path,
            "source_accession_and_license.csv",
            source_catalog_contract,
        )
        errors.extend(catalog_errors)
        if catalog_fields != SOURCE_CATALOG_FIELDS:
            errors.append("source accession and licence catalog fields mismatch")
        table_fields["source_accession_and_license.csv"] = catalog_fields

    inventory_errors, inventory = _inventory_contract(inventory_rows)
    errors.extend(inventory_errors)
    source_ids = [row.get("dataset_id", "").strip() for row in source_catalog_rows]
    if (
        len(source_ids) != len(set(source_ids))
        or "" in source_ids
        or set(source_ids) != set(inventory)
    ):
        errors.append("source catalog and inventory dataset IDs do not match")
    expected_checkpoint_paths = {
        item.path for item in inventory.values()
    }
    observed_checkpoint_paths = {
        relative
        for relative in files
        if relative.startswith("checkpoints/")
    }
    if expected_checkpoint_paths != observed_checkpoint_paths:
        errors.append("inventory and checkpoint file sets do not match")
    for dataset_id, checkpoint_contract in inventory.items():
        errors.extend(
            _validate_checkpoint(deposit_root, dataset_id, checkpoint_contract)
        )

    dictionary_path = deposit_root / "schema/data_dictionary.csv"
    if dictionary_path.is_file():
        errors.extend(_validate_dictionary(dictionary_path, table_fields))
    manifest_path = deposit_root / "release_manifest.json"
    if manifest_path.is_file():
        payload_paths = {
            *(f"tables/{name}" for name in contracts),
            *observed_checkpoint_paths,
            "catalog/source_accession_and_license.csv",
            "schema/data_dictionary.csv",
            "DATA_LICENSE",
            "LICENSE",
        }
        errors.extend(
            _validate_manifest(
                deposit_root,
                manifest_path,
                payload_paths,
                len(contracts),
                len(observed_checkpoint_paths),
            )
        )
    if (deposit_root / "checksums.sha256").is_file():
        errors.extend(_validate_checksums(deposit_root, files))

    text_scan_paths = {
        "README.md",
        "DATA_LICENSE",
        "LICENSE",
        "release_manifest.json",
        "schema/data_dictionary.csv",
    }
    if require_validation_report:
        text_scan_paths.add("VALIDATION_REPORT.md")
    for relative in sorted(text_scan_paths):
        path = files.get(relative)
        if path is not None:
            errors.extend(_scan_text_file(path, relative))
    report = deposit_root / "VALIDATION_REPORT.md"
    if require_validation_report and report.is_file():
        try:
            if "Status: **PASS**" not in report.read_text(encoding="utf-8"):
                errors.append("VALIDATION_REPORT.md does not record PASS")
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read VALIDATION_REPORT.md: {exc}")

    return ValidationResult(
        errors=tuple(sorted(set(errors))),
        table_count=canonical_table_count,
        checkpoint_count=len(observed_checkpoint_paths),
        validated_file_count=len(files),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_formal_deposit(args.root)
    if result.ok:
        print(
            f"PASS: {result.table_count} tables, "
            f"{result.checkpoint_count} checkpoints, "
            f"{result.validated_file_count} files"
        )
        return 0
    print("Formal deposit validation failed:", file=sys.stderr)
    for error in result.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
