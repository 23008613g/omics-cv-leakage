"""Load the four cached omics sources before learned preprocessing.

Sample alignment and contrast rules match the legacy loaders; imputation,
constant filtering and variance capping are deliberately deferred to folds.
"""
from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

import gxa as gxa_loader
import loader as mtbls_loader
import mbhd as mbhd_loader
import mw as mw_loader
from isatab import MIN_FEATURES, MIN_PER_CLASS, MIN_SAMPLES, norm_name


ROOT = Path(__file__).resolve().parents[1]
TransformKind = Literal[
    "intensity",
    "counts",
    "already_scaled",
    "microbiome_counts",
    "auto_nonnegative",
]
TRANSFORM_KINDS = frozenset({
    "intensity",
    "counts",
    "already_scaled",
    "microbiome_counts",
    "auto_nonnegative",
})

PARTICIPANT_FIELDS = frozenset(
    f"{entity}{suffix}"
    for entity in ("participant", "subject", "patient", "donor", "individual")
    for suffix in ("", " id", " identifier")
)


def _freeze(value: Any) -> Any:
    """Recursively freeze metadata containers without sharing caller state."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class RawDataset:
    """An immutable, unprocessed binary-phenotype dataset.

    ``X`` may contain missing or constant columns. Arrays are read-only;
    ``groups`` exists only for an explicit participant identifier.
    """

    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    feature_ids: tuple[str, ...]
    transform_kind: TransformKind
    metadata: Mapping[str, Any]
    sample_ids: tuple[str, ...]
    groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        matrix = np.array(self.X, dtype=float, copy=True)
        labels = np.array(self.y, dtype=int, copy=True)
        feature_ids = tuple(str(value) for value in self.feature_ids)
        sample_ids = tuple(str(value) for value in self.sample_ids)
        groups = (None if self.groups is None
                  else tuple(str(value) for value in self.groups))

        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if matrix.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if labels.ndim != 1 or len(labels) != matrix.shape[0]:
            raise ValueError("y must be one-dimensional and match X rows")
        if len(feature_ids) != matrix.shape[1]:
            raise ValueError("feature_ids must match X columns")
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("feature_ids must be unique")
        if len(sample_ids) != matrix.shape[0]:
            raise ValueError("sample_ids must match X rows")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if groups is not None and len(groups) != matrix.shape[0]:
            raise ValueError("groups must be None or match X rows")
        if self.transform_kind not in TRANSFORM_KINDS:
            raise ValueError(f"Unsupported transform_kind: {self.transform_kind}")

        matrix.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "X", matrix)
        object.__setattr__(self, "y", labels)
        object.__setattr__(self, "feature_ids", feature_ids)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))


def _stable_feature_ids(values: Sequence[Any], prefix: str) -> tuple[str, ...]:
    """Make deterministic, unique IDs while retaining source row order."""
    counts: dict[str, int] = {}
    output: list[str] = []
    for position, value in enumerate(values):
        missing = pd.isna(value)
        base = "" if missing else str(value).strip()
        if not base or base.lower() == "nan":
            base = f"{prefix}:feature:{position}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        output.append(base if count == 1 else f"{base}#duplicate-{count}")
    return tuple(output)


def _source_indices(index: pd.Index) -> tuple[Any, ...]:
    return tuple(value.item() if isinstance(value, np.generic) else value
                 for value in index.tolist())


def _participant_field_name(column: Any) -> str:
    """Canonicalize a metadata field name, including ISA bracket syntax."""
    text = str(column).strip().lower()
    bracketed = re.findall(r"\[([^]]+)\]", text)
    if bracketed:
        text = bracketed[-1]
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_participant_column(column: Any) -> bool:
    return _participant_field_name(column) in PARTICIPANT_FIELDS


def _participant_column(columns: Sequence[Any]) -> Any | None:
    for column in columns:
        if _is_participant_column(column):
            return column
    return None


def _groups_from_frame(
    frame: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    sample_column: str | None = None,
) -> tuple[str, ...] | None:
    column = _participant_column(frame.columns)
    if column is None:
        return None
    if sample_column is not None:
        keyed = frame.drop_duplicates(sample_column).set_index(sample_column)
    else:
        keyed = frame
    try:
        values = keyed.loc[list(sample_ids), column]
    except KeyError:
        return None
    if values.isna().any() or values.astype(str).str.strip().isin(("", "nan")).any():
        return None
    return tuple(values.astype(str).str.strip())


def _groups_from_sdrf(
    sdrf: pd.DataFrame, sample_ids: Sequence[str]
) -> tuple[str, ...] | None:
    if not {"assay", "name", "value"}.issubset(sdrf.columns):
        return None
    names = sorted({str(value) for value in sdrf["name"].dropna()
                    if _is_participant_column(value)})
    for name in names:
        rows = sdrf[sdrf["name"].astype(str).eq(name)].drop_duplicates("assay")
        values = rows.set_index("assay")["value"]
        try:
            aligned = values.loc[list(sample_ids)]
        except KeyError:
            continue
        if aligned.isna().any():
            continue
        groups = aligned.astype(str).str.strip()
        if not groups.isin(("", "nan")).any():
            return tuple(groups)
    return None


def _first_species(samples: pd.DataFrame) -> str:
    columns = [column for column in samples.columns
               if re.search(r"organism", str(column), re.I)
               and not re.search(r"part", str(column), re.I)]
    for column in columns:
        values = samples[column].dropna().astype(str).str.strip()
        values = values[~values.isin(("", "nan"))]
        if not values.empty:
            return str(values.iloc[0])
    return ""


def _maf_feature_values(maf: pd.DataFrame, sample_columns: set[str]) -> list[Any]:
    metadata_columns = [column for column in maf.columns
                        if str(column) not in sample_columns]
    priorities = (
        r"^metabolite[_ ]identification$",
        r"^database[_ ]identifier$",
        r"^smallmolecule[_ ]id$",
        r"^feature[_ ]id$",
        r"^mass[_ ]to[_ ]charge$",
    )
    for pattern in priorities:
        column = next((candidate for candidate in metadata_columns
                       if re.search(pattern, str(candidate), re.I)), None)
        if column is not None:
            return maf[column].tolist()
    return list(maf.index)


def _raw_maf_matrix(
    maf: pd.DataFrame,
    alias: Mapping[str, str],
    *,
    prefix: str,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[Any, ...]] | None:
    hits = {str(column): alias[norm_name(column)] for column in maf.columns
            if norm_name(column) in alias}
    if len(set(hits.values())) < MIN_SAMPLES:
        return None
    sample_columns = list(hits)
    matrix = maf[sample_columns].apply(pd.to_numeric, errors="coerce").T
    matrix.index = [hits[column] for column in sample_columns]
    matrix = matrix[~matrix.index.duplicated(keep="first")]
    if matrix.shape[1] < MIN_FEATURES:
        return None
    values = _maf_feature_values(maf, set(sample_columns))
    feature_ids = _stable_feature_ids(values, prefix)
    return matrix, feature_ids, _source_indices(maf.index)


def _legacy_maf_is_usable(matrix: pd.DataFrame) -> bool:
    """Match legacy MAF choice without filtering the matrix we return.

    A few studies contain several MAFs.  The historical loader skipped a
    candidate unless at least 20 columns survived its missingness and
    constant checks.  Repeating that check only as a source-file gate keeps
    the same assay choice; the caller still receives every column of the
    chosen raw MAF, including those that failed this compatibility screen.
    """
    sufficiently_observed = matrix.notna().mean(axis=0).ge(0.5)
    screened = matrix.loc[:, sufficiently_observed]
    if screened.shape[1] < MIN_FEATURES:
        return False
    imputed = screened.fillna(screened.median(axis=0))
    return int(imputed.std(axis=0).gt(1e-12).sum()) >= MIN_FEATURES


def _load_mtbls(entry: Mapping[str, Any]) -> tuple[RawDataset | None, str]:
    study = str(entry["study"])
    samples = mtbls_loader.fetch_table(study, str(entry["sample_file"]))
    if samples is None or "Sample Name" not in samples.columns:
        return None, "no_sample_file"
    samples = samples.copy()
    samples["Sample Name"] = samples["Sample Name"].astype(str).str.strip()

    contrast = mtbls_loader.pick_contrast(
        mtbls_loader.binary_contrasts(samples)
    )
    if contrast is None:
        return None, "no_binary_phenotype"

    assay = None
    for assay_name in entry.get("assay_files", ()):  # legacy first usable assay
        candidate = mtbls_loader.fetch_table(study, str(assay_name))
        if candidate is not None and "Sample Name" in candidate.columns:
            assay = candidate.copy()
            assay["Sample Name"] = assay["Sample Name"].astype(str).str.strip()
            break
    if assay is None:
        return None, "no_assay_file"
    alias = mtbls_loader.build_alias(assay)

    maf_names = list(entry.get("maf_files", ()))
    if "Metabolite Assignment File" in assay.columns:
        declared = [name for name in assay["Metabolite Assignment File"]
                    .dropna().astype(str).unique() if name in set(maf_names)]
        maf_names = declared + [name for name in maf_names if name not in declared]

    raw = None
    maf_name = ""
    for name in maf_names:
        maf = mtbls_loader.fetch_table(study, str(name))
        if maf is None:
            continue
        candidate = _raw_maf_matrix(maf, alias, prefix=f"{study}:{name}")
        if candidate is not None and _legacy_maf_is_usable(candidate[0]):
            raw = candidate
            maf_name = str(name)
            break
    if raw is None:
        return None, "no_usable_matrix"
    matrix, feature_ids, source_index = raw

    labels = samples.set_index("Sample Name")[contrast.name].astype(str).str.strip()
    labels = labels[labels.isin((contrast.level_pos, contrast.level_neg))]
    labels = labels[~labels.index.duplicated(keep="first")]
    sample_ids = [str(sample) for sample in matrix.index if sample in labels.index]
    if len(sample_ids) < MIN_SAMPLES:
        return None, "too_few_aligned"
    y = (labels.loc[sample_ids] == contrast.level_pos).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < MIN_PER_CLASS:
        return None, "class_too_small"

    metadata = {
        "repository": "MetaboLights",
        "platform": "metabolomics",
        "assay": maf_name,
        "species": _first_species(samples),
        "phenotype": contrast.name,
        "levels": (contrast.level_pos, contrast.level_neg),
        "source_feature_index": source_index,
    }
    groups = _groups_from_frame(
        samples, sample_ids, sample_column="Sample Name"
    )
    return RawDataset(
        dataset_id=study,
        X=matrix.loc[sample_ids].to_numpy(dtype=float),
        y=y,
        feature_ids=feature_ids,
        transform_kind="intensity",
        metadata=metadata,
        sample_ids=tuple(sample_ids),
        groups=groups,
    ), "ok"


def _gxa_candidate_files(files: Sequence[str]) -> list[str]:
    suffixes = (
        "-raw-counts.tsv",
        "-normalized-expressions.tsv",
        "-tpms.tsv",
        "-fpkms.tsv",
    )
    return [name for suffix in suffixes for name in files
            if str(name).endswith(suffix)]


def _gxa_feature_values(data: pd.DataFrame, metadata_columns: list[Any]) -> list[Any]:
    priorities = (
        r"gene.*id|ensembl|design element",
        r"protein.*id|uniprot",
        r"probe.*id",
        r"id$|name$",
    )
    for pattern in priorities:
        column = next((candidate for candidate in metadata_columns
                       if re.search(pattern, str(candidate), re.I)), None)
        if column is not None:
            return data[column].tolist()
    return list(data.index)


def _read_gxa_raw_matrix(
    accession: str, files: Sequence[str]
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[Any, ...], str] | None:
    for filename in _gxa_candidate_files(files):
        path = gxa_loader.get_binary(
            f"{gxa_loader.GXA_FTP}/{accession}/{filename}",
            key=f"gxa_mat_{accession}_{filename}",
        )
        if not path:
            continue
        try:
            data = pd.read_csv(path, sep="\t", low_memory=False)
        except Exception:
            continue
        metadata_columns = [column for column in data.columns
                            if re.search(
                                r"gene|design element|probe|protein|id$|name$",
                                str(column), re.I,
                            )]
        sample_columns = [column for column in data.columns
                          if column not in metadata_columns]
        if len(sample_columns) < gxa_loader.MIN_SAMPLES:
            continue
        matrix = data[sample_columns].apply(pd.to_numeric, errors="coerce").T
        if matrix.shape[1] < gxa_loader.MIN_FEATURES:
            continue
        # Compatibility gate only: retain the historical choice when an
        # accession exposes several matrices, but return every raw column of
        # the selected table below.
        if int(matrix.notna().mean(axis=0).ge(0.5).sum()) < gxa_loader.MIN_FEATURES:
            continue
        values = _gxa_feature_values(data, metadata_columns)
        feature_ids = _stable_feature_ids(values, f"{accession}:{filename}")
        return matrix, feature_ids, _source_indices(data.index), str(filename)
    return None


def _load_gxa(entry: Mapping[str, Any]) -> tuple[RawDataset | None, str]:
    accession = str(entry["accession"])
    files = gxa_loader.list_files(accession)
    if not files:
        return None, "no_ftp_listing"
    sdrf = gxa_loader.read_sdrf(accession)
    if sdrf is None:
        return None, "no_sdrf"
    contrasts = gxa_loader.contrasts_from_sdrf(sdrf)
    if not contrasts:
        return None, "no_eligible_factor"
    name, level_pos, level_neg, _, _ = min(
        contrasts,
        key=lambda contrast: (
            abs(contrast[3] - contrast[4]) / (contrast[3] + contrast[4]),
            contrast[0],
        ),
    )
    raw = _read_gxa_raw_matrix(accession, files)
    if raw is None:
        return None, "no_matrix"
    matrix, feature_ids, source_index, filename = raw

    factors = sdrf[
        sdrf["kind"].str.lower().eq("factor") & sdrf["name"].eq(name)
    ].drop_duplicates("assay")
    labels = factors.set_index("assay")["value"]
    labels = labels[labels.isin((level_pos, level_neg))]
    sample_ids = [str(sample) for sample in matrix.index if sample in labels.index]
    if len(sample_ids) < gxa_loader.MIN_SAMPLES:
        return None, "too_few_aligned"
    y = (labels.loc[sample_ids] == level_pos).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < gxa_loader.MIN_PER_CLASS:
        return None, "class_too_small"

    if filename.endswith("-raw-counts.tsv"):
        transform_kind: TransformKind = "counts"
    else:
        # Expression Atlas non-count matrices mix already-log-scaled and
        # high-range non-negative abundances.  The historical max > 50 rule
        # must therefore be fitted on each training fold, then reused with
        # the same decision and positive floor on held-out samples.
        transform_kind = "auto_nonnegative"
    metadata = {
        "repository": "Expression Atlas",
        "platform": str(entry.get("type", "")),
        "assay": str(entry.get("type", "")),
        "species": str(entry.get("species", "")),
        "phenotype": str(name),
        "levels": (str(level_pos), str(level_neg)),
        "source_file": filename,
        "source_feature_index": source_index,
    }
    return RawDataset(
        dataset_id=accession,
        X=matrix.loc[sample_ids].to_numpy(dtype=float),
        y=y,
        feature_ids=feature_ids,
        transform_kind=transform_kind,
        metadata=metadata,
        sample_ids=tuple(sample_ids),
        groups=_groups_from_sdrf(sdrf, sample_ids),
    ), "ok"


def _load_mw(entry: Mapping[str, Any]) -> tuple[RawDataset | None, str]:
    study = str(entry["study_id"])
    factors = mw_loader.parse_factors(study)
    if factors is None or factors.empty:
        return None, "no_factors"
    matrix, analysis_id = mw_loader.parse_data(study)
    if matrix is None:
        return None, "no_matrix"

    shared = [str(sample) for sample in matrix.index if sample in factors.index]
    if len(shared) < mw_loader.MIN_SAMPLES:
        return None, "too_few_aligned"
    factors = factors.loc[shared]
    contrasts: list[tuple[str, str, str, int, int]] = []
    for column in factors.columns:
        if not mw_loader.eligible_factor(str(column)):
            continue
        counts = factors[column].dropna().astype(str).str.strip().value_counts()
        if len(counts) < 2:
            continue
        (level_pos, n_pos), (level_neg, n_neg) = list(counts.items())[:2]
        if (min(n_pos, n_neg) < mw_loader.MIN_PER_CLASS
                or n_pos + n_neg < mw_loader.MIN_SAMPLES):
            continue
        contrasts.append((str(column), str(level_pos), str(level_neg),
                          int(n_pos), int(n_neg)))
    if not contrasts:
        return None, "no_eligible_factor"
    name, level_pos, level_neg, _, _ = min(
        contrasts,
        key=lambda contrast: (
            abs(contrast[3] - contrast[4]) / (contrast[3] + contrast[4]),
            contrast[0],
        ),
    )

    labels = factors[name].astype(str).str.strip()
    labels = labels[labels.isin((level_pos, level_neg))]
    sample_ids = [sample for sample in shared if sample in labels.index]
    if len(sample_ids) < mw_loader.MIN_SAMPLES:
        return None, "too_few_aligned"
    y = (labels.loc[sample_ids] == level_pos).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < mw_loader.MIN_PER_CLASS:
        return None, "class_too_small"

    numeric = matrix.loc[sample_ids].apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < mw_loader.MIN_FEATURES:
        return None, "too_few_features"
    feature_ids = _stable_feature_ids(
        numeric.columns.tolist(), f"{study}:{analysis_id}"
    )
    metadata = {
        "repository": "Metabolomics Workbench",
        "platform": "metabolomics",
        "assay": str(analysis_id),
        "analysis_type": str(entry.get("analysis_type", "")),
        "species": str(entry.get("species", "")),
        "institute": str(entry.get("institute", "")),
        "phenotype": name,
        "levels": (level_pos, level_neg),
        "source_feature_index": tuple(range(numeric.shape[1])),
    }
    return RawDataset(
        dataset_id=study,
        X=numeric.to_numpy(dtype=float),
        y=y,
        feature_ids=feature_ids,
        transform_kind="intensity",
        metadata=metadata,
        sample_ids=tuple(sample_ids),
        groups=_groups_from_frame(factors, sample_ids),
    ), "ok"


def _read_mbhd_tables(
    entry: Mapping[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    path = mbhd_loader.fetch(dict(entry))
    if path is None:
        return None, None, "download_failed"
    try:
        with tarfile.open(path, "r:gz") as archive:
            metadata = mbhd_loader._member(archive, r"\.metadata\.txt$")
            counts = mbhd_loader._member(archive, r"rdp_assigned$")
            if counts is None:
                counts = mbhd_loader._member(archive, r"otu_table\.100\.denovo$")
    except Exception as exc:
        return None, None, f"archive_error: {type(exc).__name__}"
    if metadata is None or counts is None:
        return metadata, counts, "missing_tables"
    return metadata, counts, "ok"


def _load_mbhd(entry: Mapping[str, Any]) -> tuple[RawDataset | None, str]:
    metadata, counts, reason = _read_mbhd_tables(entry)
    if reason != "ok" or metadata is None or counts is None:
        return None, reason
    metadata = metadata.copy()
    counts = counts.copy()
    metadata.index = metadata.index.astype(str).str.strip()
    counts.columns = [str(column).strip() for column in counts.columns]

    state_column = next((column for column in metadata.columns
                         if str(column).lower() in (
                             "diseasestate", "disease_state"
                         )), None)
    if state_column is None:
        return None, "no_disease_state"
    state = metadata[state_column].astype(str).str.strip()
    healthy = state.str.lower().isin(mbhd_loader.HEALTHY)
    if healthy.sum() < mbhd_loader.MIN_PER_CLASS:
        return None, "no_control_group"
    case_counts = state[~healthy].value_counts()
    if not len(case_counts) or case_counts.iloc[0] < mbhd_loader.MIN_PER_CLASS:
        return None, "no_case_group"
    case = str(case_counts.index[0])

    retained_state = state[healthy | state.eq(case)]
    sample_by_taxon = counts.T
    sample_ids = [str(sample) for sample in sample_by_taxon.index
                  if sample in retained_state.index]
    if len(sample_ids) < mbhd_loader.MIN_SAMPLES:
        return None, "too_few_aligned"
    y = (retained_state.loc[sample_ids] == case).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < mbhd_loader.MIN_PER_CLASS:
        return None, "class_too_small"

    numeric = sample_by_taxon.loc[sample_ids].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.shape[1] < mbhd_loader.MIN_FEATURES:
        return None, "too_few_features"
    study = str(entry["key"]).replace("_results.tar.gz", "")
    feature_ids = _stable_feature_ids(numeric.columns.tolist(), study)
    dataset_metadata = {
        "repository": "MicrobiomeHD",
        "platform": "microbiome",
        "assay": "16S rRNA taxonomic counts",
        "species": "Homo sapiens",
        "phenotype": "DiseaseState",
        "levels": (case, "H"),
        "source_archive": str(entry["key"]),
        "source_feature_index": _source_indices(counts.index),
    }
    return RawDataset(
        dataset_id=study,
        X=numeric.to_numpy(dtype=float),
        y=y,
        feature_ids=feature_ids,
        transform_kind="microbiome_counts",
        metadata=dataset_metadata,
        sample_ids=tuple(sample_ids),
        groups=_groups_from_frame(metadata, sample_ids),
    ), "ok"


def _canonical_platform(platform: str, entry: Mapping[str, Any] | None = None) -> str:
    value = str(platform).strip().lower()
    aliases = {
        "mtbls": "mtbls",
        "metabolights": "mtbls",
        "metabolomics (metabolights)": "mtbls",
        "gxa": "gxa",
        "expression atlas": "gxa",
        "transcriptomics": "gxa",
        "proteomics": "gxa",
        "mw": "mw",
        "workbench": "mw",
        "metabolomics workbench": "mw",
        "metabolomics (workbench)": "mw",
        "metabolomics (wb)": "mw",
        "mbhd": "mbhd",
        "microbiomehd": "mbhd",
        "microbiome": "mbhd",
    }
    if value in aliases:
        return aliases[value]
    if value == "metabolomics" and entry is not None:
        if "study_id" in entry:
            return "mw"
        if "study" in entry:
            return "mtbls"
    raise ValueError(f"Unsupported platform: {platform}")


def load_raw(
    platform: str, entry: Mapping[str, Any]
) -> tuple[RawDataset | None, str]:
    """Load one catalogue entry without learned whole-dataset preprocessing."""
    canonical = _canonical_platform(platform, entry)
    loaders = {
        "mtbls": _load_mtbls,
        "gxa": _load_gxa,
        "mw": _load_mw,
        "mbhd": _load_mbhd,
    }
    return loaders[canonical](entry)


def entry_dataset_id(platform: str, entry: Mapping[str, Any]) -> str:
    """Return the public identifier represented by a catalogue entry."""
    canonical = _canonical_platform(platform, entry)
    if canonical == "mtbls":
        return str(entry["study"])
    if canonical == "gxa":
        return str(entry["accession"])
    if canonical == "mw":
        return str(entry["study_id"])
    return str(entry["key"]).replace("_results.tar.gz", "")


@lru_cache(maxsize=1)
def _catalog_index() -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    with (ROOT / "catalog_all.json").open(encoding="utf-8") as handle:
        mtbls_entries = json.load(handle)["studies"]
    with (ROOT / "gxa_catalog.json").open(encoding="utf-8") as handle:
        gxa_entries = json.load(handle)["experiments"]
    with (ROOT / "mw_catalog.json").open(encoding="utf-8") as handle:
        mw_entries = json.load(handle)["studies"]
    with (ROOT / "mbhd_archives.json").open(encoding="utf-8") as handle:
        mbhd_entries = json.load(handle)
    entries = {
        "mtbls": {str(entry["study"]): entry for entry in mtbls_entries},
        "gxa": {str(entry["accession"]): entry for entry in gxa_entries},
        "mw": {str(entry["study_id"]): entry for entry in mw_entries},
        "mbhd": {
            str(entry["key"]).replace("_results.tar.gz", ""): entry
            for entry in mbhd_entries
        },
    }
    return MappingProxyType({source: MappingProxyType(index)
                             for source, index in entries.items()})


def _source_from_id(dataset_id: str) -> str:
    if dataset_id.startswith("MTBLS"):
        return "mtbls"
    if dataset_id.startswith("ST"):
        return "mw"
    if dataset_id.startswith("E-"):
        return "gxa"
    return "mbhd"


def _records_from_whitelist(whitelist: Any) -> list[Any]:
    if isinstance(whitelist, pd.DataFrame):
        return whitelist.to_dict(orient="records")
    if isinstance(whitelist, (str, Path)):
        path = Path(whitelist)
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path).to_dict(orient="records")
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            return _records_from_whitelist(loaded)
        return [str(whitelist)]
    if isinstance(whitelist, Mapping):
        if any(key in whitelist for key in (
            "id", "dataset_id", "study", "study_id", "accession", "key"
        )):
            return [whitelist]
        records: list[dict[str, str]] = []
        for platform, identifiers in whitelist.items():
            for identifier in identifiers:
                records.append({"id": str(identifier), "platform": str(platform)})
        return records
    if isinstance(whitelist, (set, frozenset)):
        return sorted(whitelist, key=str)
    return list(whitelist)


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("id", "dataset_id", "study", "study_id", "accession", "key"):
        if key in record and pd.notna(record[key]):
            value = str(record[key])
            return value.replace("_results.tar.gz", "")
    raise ValueError(f"Whitelist record has no dataset identifier: {record}")


def entries_for_whitelist(
    whitelist: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve whitelist IDs to cached catalogue entries in whitelist order.

    ``whitelist`` may be an iterable of IDs/records, a DataFrame, a CSV/JSON
    path, or a mapping from platform names to ID iterables.  Missing IDs fail
    loudly so a nominal 1,102-dataset rerun cannot silently shrink.
    """
    catalog = _catalog_index()
    resolved: list[tuple[str, dict[str, Any]]] = []
    for item in _records_from_whitelist(whitelist):
        if isinstance(item, Mapping):
            dataset_id = _record_id(item)
            declared = next((item[key] for key in ("platform", "plat", "repository")
                             if key in item and pd.notna(item[key])), None)
            try:
                source = (_canonical_platform(str(declared), item)
                          if declared is not None else _source_from_id(dataset_id))
            except ValueError:
                source = _source_from_id(dataset_id)
        else:
            dataset_id = str(item)
            source = _source_from_id(dataset_id)
        try:
            entry = catalog[source][dataset_id]
        except KeyError as exc:
            raise KeyError(
                f"Whitelist dataset {dataset_id!r} is absent from the "
                f"{source} catalogue"
            ) from exc
        resolved.append((source, dict(entry)))
    return resolved


__all__ = [
    "RawDataset",
    "TransformKind",
    "entries_for_whitelist",
    "entry_dataset_id",
    "load_raw",
]
