"""Whitelist, pilot-selection and provenance helpers for the v4 runner."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform as platform_module
import re
from typing import Any, Callable, Mapping, Sequence

from nm_v4.checkpoints import sha256_file, sha256_json


CANONICAL_DATASET_COUNT = 1_102
CANONICAL_ORDERED_ID_SHA256 = (
    "10cf80718e416710fdcf089981d2d3ce3fc1be3eecbd3bb1a1efd47f43499cec"
)
CANONICAL_RECORDS_SHA256 = (
    "16dd560db5dc67bc31e7076a2ec23b8a5ba87125df3fc2f5bfe129506aa347a6"
)
PILOT_SIZE = 32
PILOT_SEED = 20_260_803
EXECUTION_DEPENDENCIES = (
    "nm_v4/run_benchmark.py",
    "nm_v4/checkpoints.py",
    "nm_v4/runner_support.py",
    "nm_v4/raw_loaders.py",
    "nm_v4/protocol.py",
    "nm_v4/protocol_estimands.py",
    "nm_v4/configuration.py",
    "nm_v4/evaluation_workspace.py",
    "nm_v4/splitting.py",
    "nm_v4/preprocessing.py",
    "nm_v4/methods.py",
    "nm_v4/stability.py",
    "gxa.py",
    "loader.py",
    "mw.py",
    "mbhd.py",
    "isatab.py",
    "fetch.py",
)
_SAFE_DATASET_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class WhitelistError(ValueError):
    """The locked 1,102-dataset input list is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class WhitelistRecord:
    dataset_id: str
    repository: str
    n_samples: int
    n_features: int
    assay_platform: str

    @property
    def p_over_n(self) -> float:
        return self.n_features / self.n_samples


def load_whitelist(
    path: Path | str,
    *,
    expected_count: int = CANONICAL_DATASET_COUNT,
) -> tuple[WhitelistRecord, ...]:
    """Read and strictly validate the manuscript's canonical whitelist."""
    if expected_count != CANONICAL_DATASET_COUNT:
        raise WhitelistError(
            "The canonical benchmark is locked to exactly 1,102 dataset IDs; "
            f"received expected_count={expected_count:,}."
        )
    source = Path(path)
    required = {"id", "repository", "plat", "n_samples", "n_features"}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise WhitelistError(
                f"Whitelist is missing required columns: {sorted(missing)}"
            )
        records: list[WhitelistRecord] = []
        for row_number, row in enumerate(reader, start=2):
            dataset_id = str(row["id"] or "").strip()
            repository = str(row["repository"] or "").strip()
            assay_platform = str(row["plat"] or "").strip().lower()
            if not dataset_id or not _SAFE_DATASET_ID.fullmatch(dataset_id):
                raise WhitelistError(
                    f"Unsafe or empty dataset ID at CSV row {row_number}: "
                    f"{dataset_id!r}"
                )
            try:
                n_samples = int(float(row["n_samples"]))
                n_features = int(float(row["n_features"]))
            except (TypeError, ValueError) as exc:
                raise WhitelistError(f"Invalid n or p at CSV row {row_number}") from exc
            if not repository or not assay_platform or n_samples < 1 or n_features < 1:
                raise WhitelistError(
                    f"Invalid repository, n, or p at CSV row {row_number}"
                )
            records.append(
                WhitelistRecord(
                    dataset_id=dataset_id,
                    repository=repository,
                    n_samples=n_samples,
                    n_features=n_features,
                    assay_platform=assay_platform,
                )
            )
    if len(records) != CANONICAL_DATASET_COUNT:
        raise WhitelistError(
            "The canonical whitelist must contain exactly 1,102 rows; "
            f"found {len(records):,}."
        )
    ids = [record.dataset_id for record in records]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for dataset_id in ids:
            if dataset_id in seen:
                duplicates.add(dataset_id)
            seen.add(dataset_id)
        raise WhitelistError(
            "The 1,102 whitelist IDs must be unique; duplicate(s): "
            + ", ".join(sorted(duplicates)[:10])
        )
    id_hash = sha256_json(ids)
    if id_hash != CANONICAL_ORDERED_ID_SHA256:
        raise WhitelistError(
            "The 1,102 rows do not match the locked canonical ID list "
            f"(ordered ID SHA256 {id_hash})."
        )
    record_hash = sha256_json(
        [
            {
                "dataset_id": record.dataset_id,
                "repository": record.repository,
                "n_samples": record.n_samples,
                "n_features": record.n_features,
            }
            for record in records
        ]
    )
    if record_hash != CANONICAL_RECORDS_SHA256:
        raise WhitelistError(
            "The whitelist does not match the locked canonical record values "
            f"for ID, repository, n and p (SHA256 {record_hash})."
        )
    return tuple(records)


def _quartiles(
    records: Sequence[WhitelistRecord],
    value: Callable[[WhitelistRecord], float],
) -> dict[str, int]:
    """Assign deterministic rank quartiles, splitting ties by dataset ID."""
    ordered = sorted(records, key=lambda record: (value(record), record.dataset_id))
    count = len(ordered)
    return {
        record.dataset_id: min(3, (rank * 4) // count)
        for rank, record in enumerate(ordered)
    }


def strata_for(
    records: Sequence[WhitelistRecord],
) -> dict[str, tuple[int, int]]:
    """Return global sample-size and p/n rank quartiles for every ID."""
    if not records:
        return {}
    n_quartiles = _quartiles(records, lambda record: float(record.n_samples))
    ratio_quartiles = _quartiles(records, lambda record: record.p_over_n)
    return {
        record.dataset_id: (
            n_quartiles[record.dataset_id],
            ratio_quartiles[record.dataset_id],
        )
        for record in records
    }


def _pilot_order_key(seed: int, record: WhitelistRecord) -> tuple[str, str]:
    digest = sha256(f"{seed}:{record.dataset_id}".encode("utf-8")).hexdigest()
    return digest, record.dataset_id


def _select_repository_pilot(
    candidates: Sequence[WhitelistRecord],
    quota: int,
    strata: Mapping[str, tuple[int, int]],
    seed: int,
) -> list[WhitelistRecord]:
    remaining = list(candidates)
    selected: list[WhitelistRecord] = []
    seen_n: set[int] = set()
    seen_ratio: set[int] = set()
    seen_joint: set[tuple[int, int]] = set()
    while remaining and len(selected) < quota:

        def selection_key(record: WhitelistRecord) -> tuple[int, int, str, str]:
            n_quartile, ratio_quartile = strata[record.dataset_id]
            new_margins = int(n_quartile not in seen_n) + int(
                ratio_quartile not in seen_ratio
            )
            new_joint = int((n_quartile, ratio_quartile) not in seen_joint)
            digest, dataset_id = _pilot_order_key(seed, record)
            return -new_margins, -new_joint, digest, dataset_id

        chosen = min(remaining, key=selection_key)
        remaining.remove(chosen)
        selected.append(chosen)
        n_quartile, ratio_quartile = strata[chosen.dataset_id]
        seen_n.add(n_quartile)
        seen_ratio.add(ratio_quartile)
        seen_joint.add((n_quartile, ratio_quartile))
    return selected


def select_pilot(
    records: Sequence[WhitelistRecord],
    *,
    size: int = PILOT_SIZE,
    seed: int = PILOT_SEED,
) -> tuple[WhitelistRecord, ...]:
    """Choose a deterministic repository-balanced, two-quartile pilot."""
    if size < 1 or size > len(records):
        raise ValueError("pilot size must be between one and whitelist size")
    repositories = sorted({record.repository for record in records})
    if size < len(repositories):
        raise ValueError("pilot is too small to include every repository")
    strata = strata_for(records)
    base, remainder = divmod(size, len(repositories))
    selected: list[WhitelistRecord] = []
    for repository_index, repository in enumerate(repositories):
        quota = base + int(repository_index < remainder)
        candidates = [record for record in records if record.repository == repository]
        if len(candidates) < quota:
            raise ValueError(
                f"repository {repository!r} has fewer than {quota} datasets"
            )
        selected.extend(_select_repository_pilot(candidates, quota, strata, seed))
    return tuple(selected)


def code_fingerprint(
    source_dir: Path,
    paths: Sequence[Path] | None = None,
) -> tuple[str, dict[str, str]]:
    """Hash every executable Python source file in the v4 pipeline."""
    sources = (
        [source_dir / name for name in EXECUTION_DEPENDENCIES]
        if paths is None
        else sorted(map(Path, paths), key=lambda path: str(path))
    )
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark execution dependency/dependencies: "
            + ", ".join(str(path) for path in missing)
        )
    root = source_dir.resolve()
    file_hashes: dict[str, str] = {}
    for path in sources:
        resolved = path.resolve()
        try:
            key = resolved.relative_to(root).as_posix()
        except ValueError:
            key = str(resolved)
        file_hashes[key] = sha256_file(path)
    return sha256_json(file_hashes), file_hashes


def package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "torch")
    versions = {"python": platform_module.python_version()}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def dataset_dimensions(dataset: Any) -> tuple[int, int]:
    shape = getattr(dataset.X, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    n_samples = len(dataset.X)
    return n_samples, len(dataset.X[0]) if n_samples else 0


def anonymized_group_codes(groups: Any) -> list[int] | None:
    """Encode groups in stable first-occurrence order without exposing IDs."""
    if groups is None:
        return None
    codes: list[int] = []
    first_seen: dict[str, int] = {}
    for raw_group in groups:
        key = str(raw_group)
        if key not in first_seen:
            first_seen[key] = len(first_seen)
        codes.append(first_seen[key])
    return codes


def _hash_float_matrix(values: Any) -> tuple[list[int], str]:
    """Hash canonical little-endian float64 rows without a full-size copy."""
    import numpy as np

    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("raw X must be two-dimensional")
    shape = [int(value) for value in matrix.shape]
    digest = sha256(sha256_json({"shape": shape, "dtype": "<f8"}).encode())
    rows_per_chunk = max(1, 1_048_576 // max(shape[1], 1))
    for start in range(0, shape[0], rows_per_chunk):
        block = np.array(
            matrix[start : start + rows_per_chunk],
            dtype="<f8",
            order="C",
            copy=True,
        )
        block[np.isnan(block)] = np.nan
        block[block == 0] = 0.0  # normalize negative zero
        digest.update(block.tobytes(order="C"))
    return shape, digest.hexdigest()


def _hash_integer_labels(values: Any) -> tuple[list[int], str]:
    import numpy as np

    labels = np.array(values, dtype="<i8", order="C", copy=True)
    if labels.ndim != 1:
        raise ValueError("raw y must be one-dimensional")
    shape = [int(labels.size)]
    digest = sha256(sha256_json({"shape": shape, "dtype": "<i8"}).encode())
    digest.update(labels.tobytes(order="C"))
    return shape, digest.hexdigest()


def dataset_fingerprint(dataset: Any) -> dict[str, Any]:
    """Fingerprint every raw input that can affect observed/permuted fits."""
    feature_ids = tuple(map(str, dataset.feature_ids))
    sample_ids = tuple(map(str, dataset.sample_ids))
    x_shape, x_hash = _hash_float_matrix(dataset.X)
    y_shape, y_hash = _hash_integer_labels(dataset.y)
    if len(feature_ids) != x_shape[1] or len(sample_ids) != x_shape[0]:
        raise ValueError("feature_ids/sample_ids do not match raw X")
    if y_shape[0] != x_shape[0]:
        raise ValueError("raw y does not match raw X rows")
    groups = getattr(dataset, "groups", None)
    group_codes = anonymized_group_codes(groups)
    if groups is not None and len(groups) != x_shape[0]:
        raise ValueError("groups do not match raw X rows")
    components = {
        "x_shape": x_shape,
        "x_float64_sha256": x_hash,
        "y_shape": y_shape,
        "y_int64_sha256": y_hash,
        "feature_ids_sha256": sha256_json(list(feature_ids)),
        "sample_ids_sha256": sha256_json(list(sample_ids)),
        "groups_sha256": (
            sha256_json(list(map(str, groups))) if groups is not None else None
        ),
        "group_codes_sha256": (
            sha256_json(group_codes) if group_codes is not None else None
        ),
        "transform_kind": str(dataset.transform_kind),
    }
    return {**components, "sha256": sha256_json(components)}


def analysis_metadata(
    dataset: Any,
    *,
    source_key: str,
    n_samples: int,
    n_features: int,
) -> dict[str, Any]:
    """Build safe analysis covariates with a pre-registered broad platform."""
    import numpy as np

    labels, counts = np.unique(np.asarray(dataset.y, dtype=int), return_counts=True)
    class_counts = {
        str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)
    }
    minority = int(counts.min()) if counts.size else 0
    majority = int(counts.max()) if counts.size else 0
    source_metadata = dataset.metadata

    def safe_text(key: str) -> str | None:
        value = source_metadata.get(key)
        if value is None or isinstance(value, (list, tuple, set, dict)):
            return None
        return str(value)

    assay_platform = safe_text("platform")
    assay = safe_text("assay")
    if source_key == "gxa":
        description = f"{assay_platform or ''} {assay or ''}".upper()
        broad_platform = (
            "proteomics" if "PROTEOMICS" in description else "transcriptomics"
        )
    elif source_key in {"mtbls", "mw"}:
        broad_platform = "metabolomics"
    elif source_key == "mbhd":
        broad_platform = "microbiome"
    else:
        raise ValueError(f"unknown source key for platform mapping: {source_key}")

    return {
        "source_key": source_key,
        "source_repository": source_key,
        "repository": safe_text("repository"),
        "assay_platform": assay_platform,
        "analysis_platform": broad_platform,
        "assay": assay,
        "species": safe_text("species"),
        "phenotype": safe_text("phenotype"),
        "n_samples": n_samples,
        "n_features_raw": n_features,
        "class_counts": class_counts,
        "minority_count": minority,
        "imbalance_ratio": (minority / majority if majority else None),
        "transform_kind": str(dataset.transform_kind),
    }


def enrich_observed_result(dataset: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach non-identifying data and feature provenance to observed output."""
    feature_ids = tuple(map(str, dataset.feature_ids))
    sample_ids = tuple(map(str, dataset.sample_ids))
    n_samples, n_features = dataset_dimensions(dataset)
    if len(feature_ids) != n_features or len(sample_ids) != n_samples:
        raise ValueError("feature_ids/sample_ids do not match the raw matrix")
    selected_indices: set[int] = set()

    def collect(folds: Any) -> None:
        if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
            return
        for fold in folds:
            if not isinstance(fold, Mapping):
                continue
            for index in fold.get("selected_features", []) or []:
                normalized = int(index)
                if normalized < 0 or normalized >= n_features:
                    raise ValueError(
                        f"selected feature index {normalized} is outside raw p"
                    )
                selected_indices.add(normalized)

    methods = result.get("methods", {})
    if isinstance(methods, Mapping):
        for method_result in methods.values():
            if not isinstance(method_result, Mapping):
                continue
            observed = method_result.get("observed", {})
            if not isinstance(observed, Mapping):
                continue
            collect(observed.get("pre_split_folds", []))
            collect(observed.get("fold_contained_folds", []))
    baseline = result.get("baseline_l1", {})
    if isinstance(baseline, Mapping):
        collect(baseline.get("folds", []))
    groups = getattr(dataset, "groups", None)
    group_codes = anonymized_group_codes(groups)
    normalized_groups = list(map(str, groups)) if groups is not None else None
    provenance = {
        "feature_ids_sha256": sha256_json(list(feature_ids)),
        "sample_ids_sha256": sha256_json(list(sample_ids)),
        "selected_feature_ids": {
            str(index): feature_ids[index] for index in sorted(selected_indices)
        },
        "groups_available": groups is not None,
        "n_groups": len(set(normalized_groups)) if normalized_groups else 0,
        "groups_sha256": (
            sha256_json(normalized_groups) if normalized_groups is not None else None
        ),
        "group_codes": group_codes,
        "group_codes_sha256": (
            sha256_json(group_codes) if group_codes is not None else None
        ),
    }
    return {**copy.deepcopy(dict(result)), "data_provenance": provenance}


__all__ = [
    "CANONICAL_DATASET_COUNT",
    "CANONICAL_ORDERED_ID_SHA256",
    "CANONICAL_RECORDS_SHA256",
    "EXECUTION_DEPENDENCIES",
    "PILOT_SEED",
    "PILOT_SIZE",
    "WhitelistError",
    "WhitelistRecord",
    "analysis_metadata",
    "anonymized_group_codes",
    "code_fingerprint",
    "dataset_dimensions",
    "dataset_fingerprint",
    "enrich_observed_result",
    "load_whitelist",
    "package_versions",
    "select_pilot",
    "strata_for",
]
