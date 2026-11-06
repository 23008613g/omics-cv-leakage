"""Shared formal-deposit contracts and disclosure-control primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    from scripts.record_contracts import CANONICAL_RECORDS, RecordContract
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from record_contracts import CANONICAL_RECORDS, RecordContract

DEFAULT_RECORDS: Mapping[str, RecordContract] = MappingProxyType(
    dict(CANONICAL_RECORDS)
)
DEFAULT_SOURCE_CATALOG_CONTRACT = RecordContract(
    "catalog/source_accession_and_license.csv",
    1_102,
    5,
    "one row per attempted dataset",
)
SOURCE_CATALOG_FIELDS = (
    "dataset_id",
    "repository",
    "source_url",
    "license_note",
    "redistribution_status",
)
COMPANION_LICENSES = ("DATA_LICENSE", "LICENSE")
HUMAN_REDACT_FIELDS = frozenset(
    {
        "evidence_text",
        "reviewer_1",
        "reviewer_1_notes",
        "reviewer_2",
        "reviewer_2_notes",
        "adjudicator",
        "adjudication_notes",
    }
)
RAW_MATRIX_KEYS = frozenset(
    {
        "x",
        "raw_x",
        "matrix",
        "raw_matrix",
        "data_matrix",
        "feature_matrix",
        "expression_matrix",
        "input_matrix",
        "sample_by_feature_matrix",
    }
)
LOCAL_PATH_PATTERN = re.compile(
    r"(?:"
    r"file://[^\s'\"<>,;]*"
    r"|(?<![:/\w])/(?:Users|home|mnt|tmp|private|Volumes)"
    r"(?:/[^\s'\"<>,;]*|(?=$|[\s'\"<>,;]))"
    r"|(?<![:/\w])/var/folders"
    r"(?:/[^\s'\"<>,;]*|(?=$|[\s'\"<>,;]))"
    r"|(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"<>,;]+"
    r"|(?<!\\)\\\\[^\\\s'\"<>,;]+\\[^\\\s'\"<>,;]+"
    r")"
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
TOKEN_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{20,}\b", re.IGNORECASE),
)
SAFE_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class FieldSchema:
    name: str
    data_type: str
    nullable: bool


@dataclass(frozen=True)
class FileStats:
    path: str
    size_bytes: int
    sha256: str
    rows: int | None = None
    fields: int | None = None
    schema: tuple[FieldSchema, ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbidden_identifier_kind(value: str) -> str | None:
    if LOCAL_PATH_PATTERN.search(value):
        return "local absolute path"
    if EMAIL_PATTERN.search(value):
        return "email address"
    if any(pattern.search(value) for pattern in TOKEN_PATTERNS):
        return "credential token"
    return None


def raw_matrix_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in RAW_MATRIX_KEYS:
                return str(key)
            nested = raw_matrix_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = raw_matrix_key(item)
            if nested is not None:
                return nested
    return None


def forbidden_json_identifier(value: Any) -> str | None:
    if isinstance(value, dict):
        for item in value.values():
            finding = forbidden_json_identifier(item)
            if finding is not None:
                return finding
    elif isinstance(value, list):
        for item in value:
            finding = forbidden_json_identifier(item)
            if finding is not None:
                return finding
    elif isinstance(value, str):
        return forbidden_identifier_kind(value)
    return None


def describe_field(field: str) -> str:
    descriptions = {
        "test_index": (
            "dataset-local integer row encoding used only within the "
            "corresponding dataset; not a participant identifier and not "
            "globally comparable."
        ),
        "group_code": (
            "optional dataset-local integer encoding of an independent-unit "
            "group; not a participant identifier and not globally comparable."
        ),
        "traceback": "Cleared diagnostic traceback in the public release.",
        "checkpoint_path": "Deposit-relative path to a sanitized checkpoint.",
        "checkpoint_sha256": "SHA-256 digest of the sanitized checkpoint.",
        "checkpoint_size_bytes": "Byte size of the sanitized checkpoint.",
    }
    return descriptions.get(
        field,
        f"{field.replace('_', ' ').capitalize()} field from the canonical record.",
    )


def write_data_dictionary(
    path: Path,
    record_stats: Sequence[tuple[FileStats, RecordContract]],
) -> FileStats:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("file", "field", "type", "nullable", "scope", "description")
    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stats, contract in record_stats:
            filename = Path(stats.path).name
            for field in stats.schema:
                writer.writerow(
                    {
                        "file": filename,
                        "field": field.name,
                        "type": field.data_type,
                        "nullable": str(field.nullable).lower(),
                        "scope": contract.scope,
                        "description": describe_field(field.name),
                    }
                )
                row_count += 1
    return FileStats(
        path="schema/data_dictionary.csv",
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        rows=row_count,
        fields=len(fields),
    )


def copy_public_text(source: Path, destination: Path, relative: str) -> FileStats:
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrote_content = False
    with (
        source.open(encoding="utf-8") as input_handle,
        destination.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, start=1):
            finding = forbidden_identifier_kind(line)
            if finding is not None:
                raise ValueError(
                    f"{relative}:{line_number}: forbidden {finding}"
                )
            output_handle.write(line)
            wrote_content = wrote_content or bool(line.strip())
    if not wrote_content:
        raise ValueError(f"{relative}: public text asset is empty")
    return FileStats(
        path=relative,
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
    )


def write_readme(path: Path, checkpoint_count: int) -> None:
    text = f"""# Supervised omics evaluation records: formal deposit

This deposit contains the 12 canonical CSV record tables and
{checkpoint_count:,} individually compressed, sanitized JSON checkpoints used
by the accompanying Scientific Data Data Descriptor. It does not redistribute
upstream raw omics matrices, participant identifiers, reviewer identities,
free-text adjudication notes, tracebacks, credentials or local machine paths.

## Layout

- `tables/`: the 12 canonical record tables.
- `checkpoints/`: one deterministic `.json.gz` checkpoint per dataset.
- `catalog/source_accession_and_license.csv`: accession, upstream URL and
  licence-boundary metadata for all attempted datasets.
- `schema/data_dictionary.csv`: field-level types, scope and descriptions.
- `DATA_LICENSE` and `LICENSE`: metadata/documentation and software licences.
- `release_manifest.json`: file hashes, sizes and record contracts.
- `checksums.sha256`: SHA-256 verification list.
- `VALIDATION_REPORT.md`: build-time contract and disclosure-control result.

## Encoding boundary

`test_index` and `group_code` are dataset-local integer encodings used for
within-dataset fold auditing. They are not global participant identifiers and
must not be joined or compared across datasets.
The source accession and licence catalog is included for independent reuse.

## Validation

From the lightweight companion repository, run:

```bash
python3 scripts/validate_formal_deposit.py --root <deposit-root>
```
"""
    path.write_text(text, encoding="utf-8")


def manifest_entry(stats: FileStats, role: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": stats.path,
        "role": role,
        "sha256": stats.sha256,
        "size_bytes": stats.size_bytes,
    }
    if stats.rows is not None:
        entry["rows"] = stats.rows
    if stats.fields is not None:
        entry["fields"] = stats.fields
    return entry


def write_manifest(
    path: Path,
    table_stats: Sequence[FileStats],
    checkpoint_stats: Mapping[str, FileStats],
    dictionary_stats: FileStats,
    source_catalog_stats: FileStats,
    license_stats: Sequence[FileStats],
) -> None:
    payload_files = [
        *(manifest_entry(item, "canonical_table") for item in table_stats),
        *(
            manifest_entry(item, "sanitized_checkpoint")
            for item in checkpoint_stats.values()
        ),
        manifest_entry(dictionary_stats, "data_dictionary"),
        manifest_entry(source_catalog_stats, "source_catalog"),
        *(manifest_entry(item, "license") for item in license_stats),
    ]
    payload = {
        "format": "supervised-omics-evaluation-records-formal-deposit",
        "format_version": "1.0",
        "table_count": len(table_stats),
        "checkpoint_count": len(checkpoint_stats),
        "payload_files": sorted(payload_files, key=lambda item: item["path"]),
        "privacy_controls": {
            "checkpoint_error_tracebacks_cleared": True,
            "failure_tracebacks_cleared": True,
            "human_review_identity_and_free_text_cleared": True,
            "local_paths_emails_tokens_rejected": True,
            "raw_matrices_excluded": True,
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def write_checksums(path: Path, root: Path) -> None:
    excluded = {"checksums.sha256", "VALIDATION_REPORT.md"}
    lines = [
        f"{sha256_file(item)}  {item.relative_to(root).as_posix()}"
        for item in iter_files(root)
        if item.relative_to(root).as_posix() not in excluded
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(
    path: Path,
    table_count: int,
    checkpoint_count: int,
    validated_file_count: int,
) -> None:
    text = f"""# Formal deposit validation report

Status: **PASS**

- Canonical CSV tables: {table_count}
- Sanitized gzip checkpoints: {checkpoint_count}
- Files covered by structural validation: {validated_file_count}
- CSV row and field contracts: PASS
- Checkpoint-to-inventory hash, size and path agreement: PASS
- Checkpoint `errors[].traceback` disclosure control: PASS
- Human-adjudication identity, notes and evidence-text disclosure control: PASS
- Local absolute path, email and credential-token scan: PASS
- Raw matrix exclusion: PASS
- Release manifest and SHA-256 coverage: PASS

`VALIDATION_REPORT.md` and `checksums.sha256` are self-describing control
files and are intentionally excluded from `checksums.sha256`; all other files
are covered by that checksum list.
"""
    path.write_text(text, encoding="utf-8")


def package_size(root: Path) -> int:
    return sum(path.stat().st_size for path in iter_files(root))
