"""Fail-closed merge of independent human-cohort reviews and adjudication.

The canonical cohort scaffold is treated as immutable.  This module validates
the complete double-review queue, requires adjudication for every disagreement
or uncertain review, and publishes a new CSV plus a hash-bound sidecar.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence


TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = 1
DEFAULT_CANONICAL_ROWS = 1_102
DEFAULT_PENDING_ROWS = 258
DECISIONS = frozenset({"include", "exclude", "needs_adjudication"})
COMPONENT_VALUES = frozenset({"yes", "no", "uncertain"})
CANONICAL_STATUSES = frozenset({"include", "exclude", "needs_manual_review"})
CANONICAL_FIELDS = (
    "id",
    "repository",
    "platform",
    "n_samples",
    "n_features",
    "n_pos",
    "n_neg",
    "phenotype",
    "levels",
    "species_evidence",
    "species_evidence_source",
    "human_status",
    "disease_like_auto",
    "intervention_exclusion_auto",
    "cell_line_exclusion_auto",
    "human_participant_auto",
    "human_participant_manual",
    "manual_disease_like",
    "manual_intervention_exclusion",
    "manual_final_status",
    "strict_observational_status",
    "automatic_decision_reason",
    "evidence_source",
    "evidence_text",
    "evidence_url",
    "reviewer_1",
    "reviewer_1_decision",
    "reviewer_1_notes",
    "reviewer_2",
    "reviewer_2_decision",
    "reviewer_2_notes",
    "adjudicator",
    "adjudication_decision",
    "adjudication_notes",
    "adjudication_date",
)
REVIEW_FIELDS = (
    "id",
    "reviewer",
    "decision",
    "human_participant_manual",
    "manual_disease_like",
    "manual_intervention_exclusion",
    "notes",
)
ADJUDICATION_FIELDS = (
    "dataset_id",
    "adjudicator",
    "adjudication_decision",
    "human_participant_manual",
    "manual_disease_like",
    "manual_intervention_exclusion",
    "adjudication_notes",
    "adjudication_date",
)
REVIEW_COMPONENT_FIELDS = (
    "human_participant_manual",
    "manual_disease_like",
    "manual_intervention_exclusion",
)
OUTPUT_AUDIT_FIELDS = tuple(
    f"reviewer_{reviewer}_{field}"
    for reviewer in (1, 2)
    for field in REVIEW_COMPONENT_FIELDS
)
OUTPUT_FIELDS = (*CANONICAL_FIELDS, *OUTPUT_AUDIT_FIELDS)
CANONICAL_MANUAL_FIELDS = (
    "human_participant_manual",
    "manual_disease_like",
    "manual_intervention_exclusion",
    "manual_final_status",
    "reviewer_1",
    "reviewer_1_decision",
    "reviewer_1_notes",
    "reviewer_2",
    "reviewer_2_decision",
    "reviewer_2_notes",
    "adjudicator",
    "adjudication_decision",
    "adjudication_notes",
    "adjudication_date",
)


class AdjudicationMergeError(ValueError):
    """The audit bundle is incomplete, inconsistent, or unsafe to merge."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact_csv(
    path: Path,
    expected_fields: Sequence[str],
    *,
    label: str,
    strip_values: bool,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise AdjudicationMergeError(f"{label} is not a readable file: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise AdjudicationMergeError(
                    f"{label} must have exact columns in order: {','.join(expected_fields)}"
                )
            rows: list[dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                if None in raw or any(value is None for value in raw.values()):
                    raise AdjudicationMergeError(
                        f"{label} row {line_number} is malformed"
                    )
                row = {
                    str(key): str(value).strip() if strip_values else str(value)
                    for key, value in raw.items()
                }
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AdjudicationMergeError(f"cannot read {label}: {path}: {exc}") from exc
    return rows


def _require_unique_ids(
    rows: Sequence[Mapping[str, str]], id_field: str, *, label: str
) -> list[str]:
    identifiers = [str(row[id_field]).strip() for row in rows]
    if any(not identifier for identifier in identifiers):
        raise AdjudicationMergeError(f"{label} contains an empty dataset ID")
    if len(set(identifiers)) != len(identifiers):
        raise AdjudicationMergeError(f"{label} contains a duplicate dataset ID")
    return identifiers


def _validate_canonical(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_canonical_rows: int,
    expected_pending_rows: int,
) -> tuple[list[str], list[str]]:
    if len(rows) != expected_canonical_rows:
        raise AdjudicationMergeError(
            f"canonical row count is {len(rows)}, expected {expected_canonical_rows}"
        )
    identifiers = _require_unique_ids(rows, "id", label="canonical CSV")
    invalid_status = sorted(
        {
            str(row["strict_observational_status"]).strip()
            for row in rows
            if str(row["strict_observational_status"]).strip()
            not in CANONICAL_STATUSES
        }
    )
    if invalid_status:
        raise AdjudicationMergeError(
            f"canonical CSV contains invalid strict status: {invalid_status}"
        )
    pending = [
        str(row["id"]).strip()
        for row in rows
        if str(row["strict_observational_status"]).strip()
        == "needs_manual_review"
    ]
    if len(pending) != expected_pending_rows:
        raise AdjudicationMergeError(
            f"canonical pending row count is {len(pending)}, expected {expected_pending_rows}"
        )
    for row in rows:
        if str(row["id"]).strip() not in set(pending):
            continue
        if any(str(row[field]).strip() for field in CANONICAL_MANUAL_FIELDS):
            raise AdjudicationMergeError(
                f"pending canonical row {row['id']} already contains manual audit data"
            )
    return identifiers, pending


def _validate_review_rows(
    rows: Sequence[Mapping[str, str]], *, label: str
) -> tuple[list[str], str]:
    identifiers = _require_unique_ids(rows, "id", label=label)
    if not rows:
        raise AdjudicationMergeError(f"{label} must not be empty")
    reviewers = {str(row["reviewer"]).strip() for row in rows}
    if "" in reviewers or len(reviewers) != 1:
        raise AdjudicationMergeError(
            f"{label} must contain one non-empty identity for independent reviewer identities"
        )
    for row in rows:
        decision = str(row["decision"]).strip()
        if decision not in DECISIONS:
            raise AdjudicationMergeError(
                f"{label} has invalid decision for {row['id']}: {decision!r}"
            )
        for field in REVIEW_COMPONENT_FIELDS:
            value = str(row[field]).strip()
            if value not in COMPONENT_VALUES:
                raise AdjudicationMergeError(
                    f"{label} has invalid {field} for {row['id']}: {value!r}"
                )
        if not str(row["notes"]).strip():
            raise AdjudicationMergeError(
                f"{label} requires non-empty notes for {row['id']}"
            )
    return identifiers, next(iter(reviewers))


def _load_review_pairs(
    reviewer_1_csvs: Sequence[Path],
    reviewer_2_csvs: Sequence[Path],
    *,
    pending_ids: Sequence[str],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[dict[str, object]]]:
    if len(reviewer_1_csvs) != 3 or len(reviewer_2_csvs) != 3:
        raise AdjudicationMergeError(
            "review inputs require exactly three reviewer-1 and three reviewer-2 CSVs"
        )
    combined_1: dict[str, dict[str, str]] = {}
    combined_2: dict[str, dict[str, str]] = {}
    provenance: list[dict[str, object]] = []
    for chunk, (path_1, path_2) in enumerate(
        zip(reviewer_1_csvs, reviewer_2_csvs, strict=True), start=1
    ):
        rows_1 = _read_exact_csv(
            path_1, REVIEW_FIELDS, label=f"reviewer 1 chunk {chunk}", strip_values=True
        )
        rows_2 = _read_exact_csv(
            path_2, REVIEW_FIELDS, label=f"reviewer 2 chunk {chunk}", strip_values=True
        )
        ids_1, identity_1 = _validate_review_rows(
            rows_1, label=f"reviewer 1 chunk {chunk}"
        )
        ids_2, identity_2 = _validate_review_rows(
            rows_2, label=f"reviewer 2 chunk {chunk}"
        )
        if ids_1 != ids_2:
            raise AdjudicationMergeError(
                f"review pair {chunk} must contain the same IDs in the same order"
            )
        if identity_1 == identity_2:
            raise AdjudicationMergeError(
                f"review pair {chunk} must use independent reviewer identities"
            )
        overlap_1 = set(combined_1).intersection(ids_1)
        overlap_2 = set(combined_2).intersection(ids_2)
        if overlap_1 or overlap_2:
            repeated = sorted(overlap_1.union(overlap_2))
            raise AdjudicationMergeError(
                f"review chunks contain a duplicate dataset ID: {repeated}"
            )
        combined_1.update((row["id"], dict(row)) for row in rows_1)
        combined_2.update((row["id"], dict(row)) for row in rows_2)
        provenance.extend(
            (
                {
                    "key": f"reviewer_1_chunk_{chunk}",
                    "path": str(path_1.resolve()),
                    "row_count": len(rows_1),
                    "reviewer": identity_1,
                },
                {
                    "key": f"reviewer_2_chunk_{chunk}",
                    "path": str(path_2.resolve()),
                    "row_count": len(rows_2),
                    "reviewer": identity_2,
                },
            )
        )
    if set(combined_1) != set(pending_ids) or set(combined_2) != set(pending_ids):
        missing = sorted(set(pending_ids).difference(combined_1).union(set(pending_ids).difference(combined_2)))
        extra = sorted(set(combined_1).union(combined_2).difference(pending_ids))
        raise AdjudicationMergeError(
            "review files do not provide complete pending coverage; "
            f"missing={missing}, extra={extra}"
        )
    return combined_1, combined_2, provenance


def _components(row: Mapping[str, str]) -> tuple[str, str, str]:
    return tuple(str(row[field]).strip() for field in REVIEW_COMPONENT_FIELDS)  # type: ignore[return-value]


def _requires_adjudication(
    review_1: Mapping[str, str], review_2: Mapping[str, str]
) -> bool:
    decisions = (review_1["decision"], review_2["decision"])
    return (
        decisions[0] != decisions[1]
        or "needs_adjudication" in decisions
        or _components(review_1) != _components(review_2)
    )


def _validate_adjudication_rows(
    rows: Sequence[Mapping[str, str]], *, label: str
) -> list[str]:
    identifiers = _require_unique_ids(rows, "dataset_id", label=label)
    for row in rows:
        if not str(row["adjudicator"]).strip():
            raise AdjudicationMergeError(
                f"{label} requires a non-empty adjudicator for {row['dataset_id']}"
            )
        decision = str(row["adjudication_decision"]).strip()
        if decision not in DECISIONS:
            raise AdjudicationMergeError(
                f"{label} has invalid decision for {row['dataset_id']}: {decision!r}"
            )
        for field in REVIEW_COMPONENT_FIELDS:
            value = str(row[field]).strip()
            if value not in COMPONENT_VALUES:
                raise AdjudicationMergeError(
                    f"{label} has invalid {field} for {row['dataset_id']}: {value!r}"
                )
        if not str(row["adjudication_notes"]).strip():
            raise AdjudicationMergeError(
                f"{label} requires non-empty adjudication notes for {row['dataset_id']}"
            )
        try:
            date.fromisoformat(str(row["adjudication_date"]).strip())
        except ValueError as exc:
            raise AdjudicationMergeError(
                f"{label} requires an ISO date for {row['dataset_id']}"
            ) from exc
    return identifiers


def _load_adjudications(
    paths: Sequence[Path], *, required_ids: set[str]
) -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
    if len(paths) != 2:
        raise AdjudicationMergeError("adjudication inputs require exactly two CSVs")
    combined: dict[str, dict[str, str]] = {}
    provenance: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        rows = _read_exact_csv(
            path,
            ADJUDICATION_FIELDS,
            label=f"adjudication {index}",
            strip_values=True,
        )
        identifiers = _validate_adjudication_rows(
            rows, label=f"adjudication {index}"
        )
        overlap = set(combined).intersection(identifiers)
        if overlap:
            raise AdjudicationMergeError(
                f"adjudication files contain a duplicate dataset ID: {sorted(overlap)}"
            )
        combined.update((row["dataset_id"], dict(row)) for row in rows)
        provenance.append(
            {
                "key": f"adjudication_{index}",
                "path": str(path.resolve()),
                "row_count": len(rows),
            }
        )
    if set(combined) != required_ids:
        missing = sorted(required_ids.difference(combined))
        extra = sorted(set(combined).difference(required_ids))
        raise AdjudicationMergeError(
            "adjudication files must exactly cover all required IDs; "
            f"missing={missing}, extra={extra}"
        )
    return combined, provenance


def _merged_pending_row(
    original: Mapping[str, str],
    review_1: Mapping[str, str],
    review_2: Mapping[str, str],
    adjudication: Mapping[str, str] | None,
) -> dict[str, str]:
    merged = dict(original)
    merged.update(
        {
            "reviewer_1": review_1["reviewer"],
            "reviewer_1_decision": review_1["decision"],
            "reviewer_1_notes": review_1["notes"],
            "reviewer_2": review_2["reviewer"],
            "reviewer_2_decision": review_2["decision"],
            "reviewer_2_notes": review_2["notes"],
        }
    )
    for reviewer_number, review in ((1, review_1), (2, review_2)):
        for field in REVIEW_COMPONENT_FIELDS:
            merged[f"reviewer_{reviewer_number}_{field}"] = review[field]
    resolution = adjudication or review_1
    for field in REVIEW_COMPONENT_FIELDS:
        merged[field] = resolution[field]
    if adjudication is None:
        merged["manual_final_status"] = review_1["decision"]
    else:
        merged.update(
            {
                "manual_final_status": adjudication["adjudication_decision"],
                "adjudicator": adjudication["adjudicator"],
                "adjudication_decision": adjudication["adjudication_decision"],
                "adjudication_notes": adjudication["adjudication_notes"],
                "adjudication_date": adjudication["adjudication_date"],
            }
        )
    return {field: str(merged.get(field, "")) for field in OUTPUT_FIELDS}


def _write_staged_csv(
    directory: Path, rows: Iterable[Mapping[str, str]]
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=".human-cohort.",
        suffix=".csv.tmp",
        dir=directory,
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return staged


def _write_staged_manifest(directory: Path, payload: Mapping[str, object]) -> Path:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=".human-cohort.",
        suffix=".manifest.tmp",
        dir=directory,
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return staged


def _publish_pair(
    staged_csv: Path,
    output_csv: Path,
    staged_manifest: Path,
    output_manifest: Path,
) -> None:
    published: list[tuple[Path, Path]] = []
    try:
        for staged, destination in (
            (staged_csv, output_csv),
            (staged_manifest, output_manifest),
        ):
            os.link(staged, destination)
            published.append((staged, destination))
        directory_fd = os.open(output_csv.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for staged, destination in reversed(published):
            try:
                if os.path.samefile(staged, destination):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for staged in (staged_csv, staged_manifest):
            try:
                staged.unlink()
            except FileNotFoundError:
                pass


def _effective_status(row: Mapping[str, str]) -> str:
    adjudication = str(row["adjudication_decision"]).strip()
    manual = str(row["manual_final_status"]).strip()
    automatic = str(row["strict_observational_status"]).strip()
    if adjudication in DECISIONS:
        return adjudication
    if manual in DECISIONS:
        return manual
    return automatic


def merge_human_cohort_adjudication(
    *,
    canonical_csv: Path | str,
    reviewer_1_csvs: Sequence[Path | str],
    reviewer_2_csvs: Sequence[Path | str],
    adjudication_csvs: Sequence[Path | str],
    output_csv: Path | str,
    expected_canonical_rows: int = DEFAULT_CANONICAL_ROWS,
    expected_pending_rows: int = DEFAULT_PENDING_ROWS,
) -> dict[str, Path]:
    """Validate and atomically publish a new adjudicated cohort audit bundle."""

    canonical_path = Path(canonical_csv)
    review_1_paths = tuple(Path(path) for path in reviewer_1_csvs)
    review_2_paths = tuple(Path(path) for path in reviewer_2_csvs)
    adjudication_paths = tuple(Path(path) for path in adjudication_csvs)
    destination = Path(output_csv)
    manifest_path = destination.with_suffix(".manifest.json")
    input_paths = (
        canonical_path,
        *review_1_paths,
        *review_2_paths,
        *adjudication_paths,
    )
    resolved_inputs = [path.resolve() for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise AdjudicationMergeError("every audit input path must be unique")
    if destination.resolve() in resolved_inputs or manifest_path.resolve() in resolved_inputs:
        raise AdjudicationMergeError(
            "output CSV and manifest must be different from every input"
        )
    if destination.resolve() == manifest_path.resolve():
        raise AdjudicationMergeError("output CSV and manifest paths must differ")
    if destination.exists() or manifest_path.exists():
        occupied = destination if destination.exists() else manifest_path
        raise FileExistsError(f"refusing to overwrite existing output: {occupied}")
    if expected_canonical_rows < 1 or expected_pending_rows < 1:
        raise AdjudicationMergeError("expected row counts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)

    initial_hashes = {path.resolve(): _sha256_file(path) for path in input_paths}
    canonical_rows = _read_exact_csv(
        canonical_path,
        CANONICAL_FIELDS,
        label="canonical CSV",
        strip_values=False,
    )
    ordered_ids, pending_ids = _validate_canonical(
        canonical_rows,
        expected_canonical_rows=expected_canonical_rows,
        expected_pending_rows=expected_pending_rows,
    )
    review_1, review_2, reviewer_provenance = _load_review_pairs(
        review_1_paths, review_2_paths, pending_ids=pending_ids
    )
    required_ids = {
        dataset_id
        for dataset_id in pending_ids
        if _requires_adjudication(review_1[dataset_id], review_2[dataset_id])
    }
    adjudications, adjudication_provenance = _load_adjudications(
        adjudication_paths, required_ids=required_ids
    )

    merged_rows: list[dict[str, str]] = []
    pending_set = set(pending_ids)
    for original in canonical_rows:
        dataset_id = str(original["id"]).strip()
        if dataset_id not in pending_set:
            merged_rows.append(
                {field: str(original.get(field, "")) for field in OUTPUT_FIELDS}
            )
            continue
        merged_rows.append(
            _merged_pending_row(
                original,
                review_1[dataset_id],
                review_2[dataset_id],
                adjudications.get(dataset_id),
            )
        )
    if len(merged_rows) != expected_canonical_rows:
        raise AssertionError("merged row count changed unexpectedly")
    if [row["id"] for row in merged_rows] != ordered_ids:
        raise AssertionError("merged row order changed unexpectedly")

    effective_counts = {decision: 0 for decision in DECISIONS}
    for row in merged_rows:
        effective = _effective_status(row)
        if effective == "needs_manual_review":
            effective = "needs_adjudication"
        if effective not in effective_counts:
            raise AdjudicationMergeError(
                f"merged row {row['id']} has invalid effective status: {effective!r}"
            )
        effective_counts[effective] += 1

    staged_csv: Path | None = None
    staged_manifest: Path | None = None
    try:
        staged_csv = _write_staged_csv(destination.parent, merged_rows)
        output_hash = _sha256_file(staged_csv)
        input_sha256 = {
            "canonical": initial_hashes[canonical_path.resolve()],
            **{
                f"reviewer_1_chunk_{index}": initial_hashes[path.resolve()]
                for index, path in enumerate(review_1_paths, start=1)
            },
            **{
                f"reviewer_2_chunk_{index}": initial_hashes[path.resolve()]
                for index, path in enumerate(review_2_paths, start=1)
            },
            **{
                f"adjudication_{index}": initial_hashes[path.resolve()]
                for index, path in enumerate(adjudication_paths, start=1)
            },
        }
        for entry in (*reviewer_provenance, *adjudication_provenance):
            entry["sha256"] = input_sha256[str(entry["key"])]
        code_hash = _sha256_file(Path(__file__).resolve())
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "output_file": destination.name,
            "output_sha256": output_hash,
            "row_count": len(merged_rows),
            "pending_input_count": len(pending_ids),
            "reviewed_count": len(pending_ids),
            "reviewer1_coverage": len(review_1),
            "reviewer2_coverage": len(review_2),
            "adjudicated_count": len(adjudications),
            "final_status_counts": dict(sorted(effective_counts.items())),
            "input_sha256": input_sha256,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_sha256": code_hash,
            "output_csv_name": destination.name,
            "output_csv_sha256": output_hash,
            "base_input_path": str(canonical_path.resolve()),
            "base_input_sha256": initial_hashes[canonical_path.resolve()],
            "ordered_ids_sha256": sha256(
                ("\n".join(ordered_ids) + "\n").encode("utf-8")
            ).hexdigest(),
            "counts": {
                "canonical_rows": len(merged_rows),
                "pending_rows": len(pending_ids),
                "automatic_nonpending": len(merged_rows) - len(pending_ids),
                "consensus_without_adjudication": len(pending_ids)
                - len(adjudications),
                "adjudication_required": len(adjudications),
                "effective_include": effective_counts["include"],
                "effective_exclude": effective_counts["exclude"],
                "effective_unresolved": effective_counts["needs_adjudication"],
            },
            "reviewer_inputs": reviewer_provenance,
            "adjudication_inputs": adjudication_provenance,
        }
        staged_manifest = _write_staged_manifest(destination.parent, manifest)
        current_hashes = {
            path.resolve(): _sha256_file(path) for path in input_paths
        }
        if current_hashes != initial_hashes:
            raise AdjudicationMergeError(
                "one or more audit inputs changed during merge; no output published"
            )
        if destination.exists() or manifest_path.exists():
            occupied = destination if destination.exists() else manifest_path
            raise FileExistsError(f"refusing to overwrite existing output: {occupied}")
        _publish_pair(staged_csv, destination, staged_manifest, manifest_path)
        staged_csv = None
        staged_manifest = None
    finally:
        for staged in (staged_csv, staged_manifest):
            if staged is not None:
                try:
                    staged.unlink()
                except FileNotFoundError:
                    pass
    return {"csv": destination, "manifest": manifest_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge the complete double-reviewed human cohort queue."
    )
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--reviewer-1", required=True, type=Path, nargs=3)
    parser.add_argument("--reviewer-2", required=True, type=Path, nargs=3)
    parser.add_argument("--adjudication", required=True, type=Path, nargs=2)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-canonical-rows", type=int, default=DEFAULT_CANONICAL_ROWS
    )
    parser.add_argument("--expected-pending-rows", type=int, default=DEFAULT_PENDING_ROWS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = merge_human_cohort_adjudication(
        canonical_csv=args.canonical,
        reviewer_1_csvs=args.reviewer_1,
        reviewer_2_csvs=args.reviewer_2,
        adjudication_csvs=args.adjudication,
        output_csv=args.output,
        expected_canonical_rows=args.expected_canonical_rows,
        expected_pending_rows=args.expected_pending_rows,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
