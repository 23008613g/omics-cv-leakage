"""Resumable runner for the 1,102-dataset Nature Methods benchmark.

Each process-pool job loads one raw dataset exactly once, evaluates observed
labels, and evaluates deterministic permutation indices in independently
mergeable blocks.  Every completed block is atomically checkpointed.  This
module never reads or writes the legacy ``arms*.json`` files.
"""

from __future__ import annotations

import os

# Set before importing anything that can transitively import NumPy.
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in BLAS_THREAD_VARIABLES:
    os.environ[_thread_variable] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402

# Re-export the tested checkpoint and selection API from this orchestration
# module so existing callers do not change when helpers are split by concern.
from nm_v4.checkpoints import (  # noqa: E402
    CheckpointMismatchError,
    CheckpointStore,
    SCHEMA_VERSION,
    atomic_write_gzip_json as atomic_write_gzip_json,
    atomic_write_json,
    checkpoint_complete,
    completed_permutation_indices,
    extend_target,
    merge_observed,
    merge_permutation_block,
    missing_permutation_blocks,
    new_checkpoint,
    read_gzip_json as read_gzip_json,
    record_error,
    sha256_file,
    sha256_json,
    summary_from_state,
    utc_now,
)
from nm_v4.runner_support import (  # noqa: E402
    CANONICAL_DATASET_COUNT,
    EXECUTION_DEPENDENCIES as EXECUTION_DEPENDENCIES,
    PILOT_SEED as PILOT_SEED,
    PILOT_SIZE as PILOT_SIZE,
    WhitelistError,
    WhitelistRecord,
    analysis_metadata,
    code_fingerprint as _code_fingerprint,
    dataset_dimensions,
    dataset_fingerprint as dataset_fingerprint,
    enrich_observed_result as enrich_observed_result,
    load_whitelist as _load_whitelist,
    package_versions,
    select_pilot,
    strata_for as strata_for,
)


DEFAULT_WORKERS = 6
DEFAULT_PERMUTATIONS = 199
PILOT_PERMUTATIONS = 19
DEFAULT_BLOCK_SIZE = 50
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHITELIST = (
    ROOT.parent
    / "manuscript_B"
    / "Nature_Methods_Review_Archive_v2"
    / "measurements"
    / "per_dataset_measurements.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "nm_v4_results"


@dataclass(frozen=True, slots=True)
class DatasetJob:
    dataset_id: str
    platform: str
    entry: dict[str, Any]
    output_dir: str
    protocol_config: dict[str, Any]
    analysis_config_sha256: str
    target_config_sha256: str
    code_sha256: str
    environment_sha256: str
    permutation_plan_sha256: str
    assay_platform: str | None = None
    target_permutations: int = DEFAULT_PERMUTATIONS
    block_size: int = DEFAULT_BLOCK_SIZE

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if self.target_permutations < 0:
            raise ValueError("target_permutations must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")


def load_whitelist(
    path: Path | str = DEFAULT_WHITELIST,
    *,
    expected_count: int = CANONICAL_DATASET_COUNT,
) -> tuple[WhitelistRecord, ...]:
    return _load_whitelist(path, expected_count=expected_count)


def code_fingerprint(
    paths: Sequence[Path] | None = None,
    *,
    source_dir: Path | None = None,
) -> tuple[str, dict[str, str]]:
    directory = ROOT if source_dir is None else source_dir
    return _code_fingerprint(directory, paths)


def _default_loader(platform: str, entry: Mapping[str, Any]) -> tuple[Any, str]:
    from nm_v4 import raw_loaders

    return raw_loaders.load_raw(platform, entry)


def _protocol_config(job: DatasetJob) -> Any:
    from nm_v4.protocol import ProtocolConfig

    return ProtocolConfig(**job.protocol_config)


def _default_observed_evaluator(dataset: Any, job: DatasetJob) -> dict[str, Any]:
    from nm_v4.protocol import evaluate_observed

    return evaluate_observed(
        dataset.X,
        dataset.y,
        dataset_id=job.dataset_id,
        config=_protocol_config(job),
        transform_kind=dataset.transform_kind,
        groups=dataset.groups,
    )


def _default_permutation_evaluator(
    dataset: Any, job: DatasetJob, indices: list[int]
) -> dict[str, Any]:
    from nm_v4.protocol import evaluate_permutation_block

    return evaluate_permutation_block(
        dataset.X,
        dataset.y,
        dataset_id=job.dataset_id,
        config=_protocol_config(job),
        permutation_indices=indices,
        transform_kind=dataset.transform_kind,
        groups=dataset.groups,
    )


def _mismatch_summary(
    job: DatasetJob,
    message: str,
    completed: int = 0,
) -> dict[str, Any]:
    return {
        "dataset_id": job.dataset_id,
        "status": "checkpoint_mismatch",
        "message": message,
        "completed_permutations": completed,
        "target_permutations": job.target_permutations,
    }


def execute_dataset(
    job: DatasetJob,
    *,
    loader: Callable[[str, Mapping[str, Any]], tuple[Any, str]] | None = None,
    observed_evaluator: Callable[[Any, DatasetJob], Mapping[str, Any]] | None = None,
    permutation_evaluator: Callable[[Any, DatasetJob, list[int]], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Run or resume one dataset; suitable for a process-pool worker."""
    store = CheckpointStore(
        job.output_dir,
        analysis_config_sha256=job.analysis_config_sha256,
        code_sha256=job.code_sha256,
        environment_sha256=job.environment_sha256,
    )
    with store.dataset_lock(job.dataset_id):
        return _execute_dataset_locked(
            job,
            store,
            loader=loader,
            observed_evaluator=observed_evaluator,
            permutation_evaluator=permutation_evaluator,
        )


def _execute_dataset_locked(
    job: DatasetJob,
    store: CheckpointStore,
    *,
    loader: Callable[[str, Mapping[str, Any]], tuple[Any, str]] | None = None,
    observed_evaluator: Callable[[Any, DatasetJob], Mapping[str, Any]] | None = None,
    permutation_evaluator: Callable[[Any, DatasetJob, list[int]], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    load_once = loader or _default_loader
    evaluate_observed_once = observed_evaluator or _default_observed_evaluator
    evaluate_block = permutation_evaluator or _default_permutation_evaluator
    try:
        state = store.load(job.dataset_id, job.permutation_plan_sha256)
    except CheckpointMismatchError as exc:
        return _mismatch_summary(job, str(exc))
    if state is None:
        state = new_checkpoint(
            dataset_id=job.dataset_id,
            analysis_config_sha256=job.analysis_config_sha256,
            target_config_sha256=job.target_config_sha256,
            code_sha256=job.code_sha256,
            environment_sha256=job.environment_sha256,
            permutation_plan_sha256=job.permutation_plan_sha256,
            target_permutations=job.target_permutations,
            metadata={"source_key": job.platform},
        )
        store.save(state)
    else:
        try:
            state = extend_target(
                state, job.target_permutations, job.target_config_sha256
            )
        except CheckpointMismatchError as exc:
            return _mismatch_summary(
                job, str(exc), len(completed_permutation_indices(state))
            )
    try:
        dataset, reason = load_once(job.platform, job.entry)
    except Exception as exc:  # noqa: BLE001 - unattended errors are recorded
        state = record_error(state, status="load_error", phase="load", error=exc)
        store.save(state)
        return summary_from_state(state)
    if dataset is None or reason != "ok":
        state = record_error(
            state, status="load_error", phase="load", error=str(reason)
        )
        store.save(state)
        return summary_from_state(state)
    if str(getattr(dataset, "dataset_id", "")) != job.dataset_id:
        state = record_error(
            state,
            status="load_error",
            phase="load",
            error=(
                f"loader returned dataset ID {getattr(dataset, 'dataset_id', None)!r} "
                f"for requested {job.dataset_id!r}"
            ),
        )
        store.save(state)
        return summary_from_state(state)

    try:
        fingerprint = dataset_fingerprint(dataset)
    except Exception as exc:  # noqa: BLE001
        state = record_error(
            state, status="data_error", phase="data_fingerprint", error=exc
        )
        store.save(state)
        return summary_from_state(state)
    previous_fingerprint = state.get("data_fingerprint")
    if previous_fingerprint is not None and previous_fingerprint != fingerprint:
        state = record_error(
            state,
            status="data_mismatch",
            phase="data_fingerprint",
            error=(
                "raw data fingerprint changed; refusing to combine cached "
                f"{previous_fingerprint.get('sha256')} with current "
                f"{fingerprint.get('sha256')}"
            ),
        )
        store.save(state)
        return summary_from_state(state)

    n_samples, n_features = dataset_dimensions(dataset)
    state = copy.deepcopy(state)
    state["data_fingerprint"] = fingerprint
    state["metadata"] = {
        **state.get("metadata", {}),
        **analysis_metadata(
            dataset,
            source_key=job.platform,
            n_samples=n_samples,
            n_features=n_features,
        ),
    }
    if checkpoint_complete(state):
        state["status"] = "ok"
        state["updated_at"] = utc_now()
        store.save(state)
        return summary_from_state(state)
    state["status"] = "running"
    state["updated_at"] = utc_now()
    store.save(state)

    if state.get("observed") is None:
        try:
            observed_result = enrich_observed_result(
                dataset, evaluate_observed_once(dataset, job)
            )
            state = merge_observed(state, observed_result)
            store.save(state)
        except Exception as exc:  # noqa: BLE001
            state = record_error(
                state, status="evaluation_error", phase="observed", error=exc
            )
            store.save(state)
            return summary_from_state(state)

    missing_blocks = missing_permutation_blocks(
        completed_permutation_indices(state),
        job.target_permutations,
        job.block_size,
    )
    for indices in missing_blocks:
        phase = f"permutations:{indices[0]}-{indices[-1]}"
        try:
            state = merge_permutation_block(
                state, indices, evaluate_block(dataset, job, indices)
            )
            store.save(state)
        except Exception as exc:  # noqa: BLE001
            state = record_error(
                state, status="evaluation_error", phase=phase, error=exc
            )
            store.save(state)
            return summary_from_state(state)

    state = {
        **state,
        "status": "ok",
        "completed_at": utc_now(),
        "updated_at": utc_now(),
    }
    store.save(state)
    return summary_from_state(state)


def _worker(job: DatasetJob) -> dict[str, Any]:
    return execute_dataset(job)


def _protocol_configuration(target_permutations: int) -> tuple[dict[str, Any], str]:
    from nm_v4 import protocol

    config = protocol.ProtocolConfig(n_permutations=target_permutations)
    payload = asdict(config)
    if hasattr(protocol, "protocol_config_sha256"):
        analysis_hash = protocol.protocol_config_sha256(
            config, include_permutation_count=False
        )
    else:
        analysis_hash = sha256_json(
            {key: value for key, value in payload.items() if key != "n_permutations"}
        )
    return payload, str(analysis_hash)


def _permutation_plan_hash(
    dataset_id: str,
    protocol_config: Mapping[str, Any],
    analysis_config_sha256: str,
) -> str:
    return sha256_json(
        {
            "algorithm": "nm_v4-indexed-label-permutation-v1",
            "dataset_id": dataset_id,
            "base_seed": int(protocol_config["seed"]),
            "analysis_config_sha256": analysis_config_sha256,
        }
    )


def _parse_ids(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    parsed = [
        dataset_id.strip()
        for value in values
        for dataset_id in value.split(",")
        if dataset_id.strip()
    ]
    if not parsed:
        raise ValueError("--ids did not contain any dataset IDs")
    if len(parsed) != len(set(parsed)):
        raise ValueError("--ids contains duplicate dataset IDs")
    return parsed


def _selected_records(
    whitelist: Sequence[WhitelistRecord],
    *,
    pilot: bool,
    ids: Sequence[str] | None,
) -> tuple[WhitelistRecord, ...]:
    by_id = {record.dataset_id: record for record in whitelist}
    if ids is None:
        return select_pilot(whitelist) if pilot else tuple(whitelist)
    unknown = [dataset_id for dataset_id in ids if dataset_id not in by_id]
    if unknown:
        raise WhitelistError(
            "--ids contains IDs outside the locked whitelist: "
            + ", ".join(unknown[:10])
        )
    selected = tuple(by_id[dataset_id] for dataset_id in ids)
    if pilot:
        pilot_ids = {record.dataset_id for record in select_pilot(whitelist)}
        outside = [
            record.dataset_id
            for record in selected
            if record.dataset_id not in pilot_ids
        ]
        if outside:
            raise WhitelistError(
                "--pilot with --ids accepts only deterministic pilot IDs; "
                f"outside pilot: {', '.join(outside[:10])}"
            )
    return selected


def _build_jobs(
    selected: Sequence[WhitelistRecord],
    *,
    output_dir: Path,
    protocol_config: dict[str, Any],
    analysis_config_sha256: str,
    target_config_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    target_permutations: int,
    block_size: int,
) -> list[DatasetJob]:
    from nm_v4 import raw_loaders

    resolved = raw_loaders.entries_for_whitelist(
        [record.dataset_id for record in selected]
    )
    if len(resolved) != len(selected):
        raise RuntimeError("raw loader did not resolve every selected dataset")
    jobs: list[DatasetJob] = []
    for record, (platform, entry) in zip(selected, resolved, strict=True):
        resolved_id = raw_loaders.entry_dataset_id(platform, entry)
        if resolved_id != record.dataset_id:
            raise RuntimeError(
                f"catalogue order mismatch: expected {record.dataset_id}, "
                f"received {resolved_id}"
            )
        jobs.append(
            DatasetJob(
                dataset_id=record.dataset_id,
                platform=platform,
                entry=dict(entry),
                output_dir=str(output_dir),
                protocol_config=copy.deepcopy(protocol_config),
                analysis_config_sha256=analysis_config_sha256,
                target_config_sha256=target_config_sha256,
                code_sha256=code_sha256,
                environment_sha256=environment_sha256,
                permutation_plan_sha256=_permutation_plan_hash(
                    record.dataset_id, protocol_config, analysis_config_sha256
                ),
                assay_platform=record.assay_platform,
                target_permutations=target_permutations,
                block_size=block_size,
            )
        )
    return jobs


def _manifest(
    *,
    whitelist_path: Path,
    whitelist_sha256: str,
    selected: Sequence[WhitelistRecord],
    pilot: bool,
    protocol_config: Mapping[str, Any],
    analysis_config_sha256: str,
    target_config_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    code_files: Mapping[str, str],
    versions: Mapping[str, str],
    workers: int,
    block_size: int,
    results: Mapping[str, Mapping[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete"
            if len(results) == len(selected)
            and all(result.get("status") == "ok" for result in results.values())
            else "running_or_incomplete"
        ),
        "started_at": started_at,
        "updated_at": utc_now(),
        "pilot": bool(pilot),
        "whitelist_path": str(whitelist_path.resolve()),
        "whitelist_sha256": whitelist_sha256,
        "canonical_whitelist_count": CANONICAL_DATASET_COUNT,
        "selected_dataset_count": len(selected),
        "selected_dataset_ids": [record.dataset_id for record in selected],
        "protocol_config": copy.deepcopy(dict(protocol_config)),
        "analysis_config_sha256": analysis_config_sha256,
        "target_config_sha256": target_config_sha256,
        "code_sha256": code_sha256,
        "environment_sha256": environment_sha256,
        "code_file_sha256": dict(code_files),
        "package_versions": dict(versions),
        "workers": workers,
        "permutation_block_size": block_size,
        "datasets": dict(sorted(results.items())),
    }


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run(args: argparse.Namespace) -> int:
    whitelist_path = Path(args.whitelist)
    whitelist = load_whitelist(whitelist_path)
    selected = _selected_records(
        whitelist, pilot=bool(args.pilot), ids=_parse_ids(args.ids)
    )
    target_permutations = (
        int(args.permutations)
        if args.permutations is not None
        else (PILOT_PERMUTATIONS if args.pilot else DEFAULT_PERMUTATIONS)
    )
    if target_permutations < 0:
        raise ValueError("--permutations must be non-negative")
    if args.workers < 1 or args.block_size < 1:
        raise ValueError("--workers and --block-size must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_config, analysis_hash = _protocol_configuration(target_permutations)
    target_hash = sha256_json(protocol_config)
    code_hash, code_files = code_fingerprint()
    versions = package_versions()
    environment_hash = sha256_json(versions)
    jobs = _build_jobs(
        selected,
        output_dir=output_dir,
        protocol_config=protocol_config,
        analysis_config_sha256=analysis_hash,
        target_config_sha256=target_hash,
        code_sha256=code_hash,
        environment_sha256=environment_hash,
        target_permutations=target_permutations,
        block_size=args.block_size,
    )
    started_at = utc_now()
    results: dict[str, dict[str, Any]] = {}
    manifest_path = output_dir / "manifest.json"
    whitelist_hash = sha256_file(whitelist_path)

    def write_manifest() -> None:
        atomic_write_json(
            manifest_path,
            _manifest(
                whitelist_path=whitelist_path,
                whitelist_sha256=whitelist_hash,
                selected=selected,
                pilot=bool(args.pilot),
                protocol_config=protocol_config,
                analysis_config_sha256=analysis_hash,
                target_config_sha256=target_hash,
                code_sha256=code_hash,
                environment_sha256=environment_hash,
                code_files=code_files,
                versions=versions,
                workers=args.workers,
                block_size=args.block_size,
                results=results,
                started_at=started_at,
            ),
        )

    write_manifest()
    print(
        f"nm_v4: {len(jobs):,} datasets, {target_permutations} indexed "
        f"permutations, {args.workers} workers, output={output_dir}",
        flush=True,
    )
    start_time = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - report worker failures
                result = {
                    "dataset_id": job.dataset_id,
                    "status": "worker_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "completed_permutations": 0,
                    "target_permutations": target_permutations,
                }
            results[job.dataset_id] = result
            write_manifest()
            elapsed = time.monotonic() - start_time
            eta = (elapsed / completed) * (len(jobs) - completed)
            print(
                f"[{completed:4d}/{len(jobs):4d}] {job.dataset_id:<20} "
                f"{result['status']:<20} "
                f"perm={result.get('completed_permutations', 0):>3}/"
                f"{target_permutations:<3} ETA {_format_duration(eta)}",
                flush=True,
            )
    write_manifest()
    failed = [result for result in results.values() if result["status"] != "ok"]
    print(
        f"Finished: {len(results) - len(failed):,} ok, {len(failed):,} failed; "
        f"manifest={manifest_path}",
        flush=True,
    )
    return int(bool(failed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fold-contained five-method benchmark using atomic, "
            "incrementally resumable per-dataset checkpoints."
        )
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "run the deterministic 32-dataset stratified pilot "
            f"(default {PILOT_PERMUTATIONS} permutations)"
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=None,
        help=f"target permutations (formal default {DEFAULT_PERMUTATIONS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"dataset worker processes (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"v4 result directory (default {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="optional whitelist IDs, separated by spaces and/or commas",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"permutation checkpoint block size (default {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=DEFAULT_WHITELIST,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, WhitelistError, CheckpointMismatchError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
