"""Build a conservative, auditable human observational-cohort scaffold.

Strict inclusion requires explicit local evidence for every element. Human
review fields remain blank, inputs are read-only, and outputs are new files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STATUS_VALUES = frozenset({"include", "exclude", "needs_manual_review"})
SOURCE_URLS = {
    "gxa": "https://www.ebi.ac.uk/gxa/experiments/{accession}",
    "mtbls": "https://www.ebi.ac.uk/metabolights/{accession}",
    "mw": ("https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID={accession}"),
    "mbhd": "https://doi.org/10.5281/zenodo.569601",
}
OUTPUT_FIELDS = (
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
_UNKNOWN_SPECIES = re.compile(
    r"^(?:|unknown|unspecified|not available|not applicable|na|n/a|mixed)$",
    re.I,
)
_HUMAN_SPECIES = re.compile(r"\b(?:homo\s+sapiens|human|ncbitaxon[_:\s-]*9606)\b", re.I)
_CELL_LINE = re.compile(
    r"\bcell[ -]?line\b|\bcell culture\b|\bcultured cells?\b|"
    r"\bimmortali[sz]ed\b|\borganoid\b",
    re.I,
)
_PARTICIPANT = re.compile(
    r"\bpatients?\b|\bparticipants?\b|\bsubjects?\b|\bvolunteers?\b|"
    r"\bdonors?\b|\bcase[- ]control\b|\bclinical cohort\b|"
    r"\bhuman study subjects?\b|\bindividual present\b|"
    r"\b(?:serum|plasma|whole blood|peripheral blood|urine|stool|faeces|"
    r"feces|biopsy|cerebrospinal fluid)\b",
    re.I,
)
_DISEASE_FIELD = re.compile(r"disease(?:state)?|diagnos|patholog", re.I)
_SPECIFIC_DISEASE = re.compile(
    r"cancer|tumou?r|carcinom|leukemi|lymphom|melanom|diabet|\bobes(?:e|ity)\b|"
    r"alzheimer|parkinson|\bhiv\b|\baids\b|\bibd\b|crohn|colitis|arthritis|"
    r"asthma|\bcopd\b|sepsis|cirrhosis|\bnash\b|\bnafld\b|autis|\basd\b|"
    r"\bcrc\b|\bcdi\b|\bt1d\b|\bmhe\b|psoriasis|covid|syndrome|"
    r"infection status|infected",
    re.I,
)
_GENERIC_CASE = re.compile(r"^(?:case|disease|affected|patient)s?$", re.I)
_CLEAR_NON_DISEASE_FACTOR = re.compile(
    r"^(?:factor value\[)?\s*(?:sex|gender|genotype|strain|variant|"
    r"treatment|compound|dose|drug|stimulus|time|age|diet|exercise|"
    r"batch|site|tissue|organism part)\s*(?:\])?$",
    re.I,
)
_INTERVENTION = re.compile(
    r"\btreat(?:ment|ed|ment group)?\b|\bcompound\b|\bdose\b|\bdrug\b|"
    r"\bintervention\b|\brandomi[sz]ed\b|\bplacebo\b|\btherapy\b|"
    r"\bstim(?:ulus|uli|ulate|ulated|ulation)\b|\bchallenge(?:d)?\b|\bexposure\b|"
    r"\bpre[- ]?treatment\b|\bpost[- ]?treatment\b|\bexercise\b|"
    r"\bdiet(?:ary)?\b|\bsurgery\b|\bvaccin|\birradiat|\btransfect|"
    r"\bknock(?:down|out)\b",
    re.I,
)
_CONTROL = re.compile(
    r"^(?:h|hc|healthy|healthy control|control|controls|normal|unaffected|"
    r"non[- ]?disease|non[- ]?case)$",
    re.I,
)
_OBSERVATIONAL = re.compile(
    r"\bcase[- ]control\b|\bobservation(?:al)? cohort\b|"
    r"\bpatients?\b.{0,80}\bcontrols?\b|"
    r"\bhealthy controls?\b.{0,80}\bpatients?\b",
    re.I,
)
_BIOSPECIMEN = re.compile(
    r"\bserum\b|\bplasma\b|\bwhole blood\b|\bperipheral blood\b|"
    r"\burine\b|\bstool\b|\bfaeces\b|\bfeces\b|\bbiopsy\b|"
    r"\bcerebrospinal fluid\b",
    re.I,
)


@dataclass(frozen=True)
class EvidenceBundle:
    """Immutable evidence passed from repository readers to the classifier."""

    species_values: tuple[str, ...] = ()
    species_sources: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    fragments: tuple[str, ...] = ()
    evidence_url: str = ""
    explicit_participant: bool = False
    explicit_cell_line: bool = False
    explicit_case_control: bool = False
    explicit_intervention: bool = False
    mbhd_human_case_control: bool = False


def _clean(value: object, limit: int = 1_200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _unique(values: Iterable[object]) -> tuple[str, ...]:
    cleaned = (_clean(value) for value in values)
    return tuple(dict.fromkeys(value for value in cleaned if value))


def _repository(record: Mapping[str, str]) -> str:
    explicit = _clean(record.get("repository", "")).lower()
    if explicit in SOURCE_URLS:
        return explicit
    platform = _clean(record.get("plat") or record.get("platform", "")).lower()
    accession = _clean(record.get("id", ""))
    if "workbench" in platform or accession.startswith("ST"):
        return "mw"
    if "metabolights" in platform or accession.startswith("MTBLS"):
        return "mtbls"
    if "microbiome" in platform or not re.match(r"^(?:E-|MTBLS|ST)", accession):
        return "mbhd" if "microbiome" in platform else explicit
    if "transcript" in platform or "proteom" in platform or accession.startswith("E-"):
        return "gxa"
    return explicit


def _source_url(repository: str, accession: str) -> str:
    template = SOURCE_URLS.get(repository, "")
    return template.format(accession=accession) if template else ""


def _parse_levels(raw: object) -> tuple[str, ...]:
    text = _clean(raw)
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return _unique(parsed)
    return _unique(re.split(r"\s*(?:\||;|,|\bvs\.?\b)\s*", text, flags=re.I))


def _human_status(species_values: Sequence[str]) -> str:
    states: set[str] = set()
    for value in species_values:
        for item in re.split(r"\s*[;|]\s*", _clean(value)):
            if _UNKNOWN_SPECIES.match(item):
                states.add("uncertain")
            elif _HUMAN_SPECIES.search(item):
                states.add("human")
            else:
                states.add("nonhuman")
    if states == {"human"}:
        return "human"
    if states == {"nonhuman"}:
        return "nonhuman"
    return "uncertain"


def _disease_like(record: Mapping[str, str]) -> str:
    phenotype = _clean(record.get("phenotype", ""))
    levels = _parse_levels(record.get("levels", ""))
    if _CLEAR_NON_DISEASE_FACTOR.match(phenotype):
        return "no"
    if _SPECIFIC_DISEASE.search(" ".join((phenotype, *levels))):
        return "yes"
    if _DISEASE_FIELD.search(phenotype) and any(
        _DISEASE_FIELD.search(level) or _GENERIC_CASE.match(level) for level in levels
    ):
        return "yes"
    return "uncertain"


def _case_control_from_record(record: Mapping[str, str]) -> bool:
    phenotype = _clean(record.get("phenotype", ""))
    levels = _parse_levels(record.get("levels", ""))
    has_control = any(_CONTROL.match(level) for level in levels)
    has_case = any(
        not _CONTROL.match(level) and (_SPECIFIC_DISEASE.search(level) or _GENERIC_CASE.match(level))
        for level in levels
    )
    phenotype_support = bool(_DISEASE_FIELD.search(phenotype) or _SPECIFIC_DISEASE.search(phenotype))
    return bool(phenotype_support and has_control and has_case)


def classify_record(record: Mapping[str, str], evidence: EvidenceBundle) -> dict[str, str]:
    """Return conservative automatic fields and blank human-review fields."""

    species = _unique((*evidence.species_values, record.get("species", "")))
    human = _human_status(species)
    disease = _disease_like(record)
    chosen_contrast = " ".join((_clean(record.get("phenotype", "")), _clean(record.get("levels", ""))))
    evidence_text = " ; ".join(_unique(evidence.fragments))
    cell_line = bool(evidence.explicit_cell_line or _CELL_LINE.search(evidence_text))
    intervention = bool(evidence.explicit_intervention or _INTERVENTION.search(chosen_contrast))
    participant = bool(evidence.explicit_participant or (human == "human" and _PARTICIPANT.search(evidence_text)))
    case_control = bool(
        evidence.explicit_case_control or _case_control_from_record(record) or _OBSERVATIONAL.search(evidence_text)
    )

    participant_auto = (
        "no" if human == "nonhuman" or cell_line else "yes" if human == "human" and participant else "uncertain"
    )
    cell_line_auto = "yes" if cell_line else "no" if participant_auto == "yes" else "uncertain"

    # "no" means there is positive observational/case-control support,
    # not merely that no intervention word happened to be found.
    observational_support = case_control
    intervention_auto = "yes" if intervention else "no" if observational_support else "uncertain"

    reasons: list[str] = []
    if human == "nonhuman":
        reasons.append("nonhuman_species")
    if disease == "no":
        reasons.append("non_disease_contrast")
    if intervention_auto == "yes":
        reasons.append("intervention_contrast")
    if cell_line_auto == "yes":
        reasons.append("cell_line_or_culture")
    if participant_auto == "no":
        reasons.append("not_human_participants")

    hard_exclusion = bool(reasons)
    repository = _repository(record)
    ready = (
        human == "human"
        and disease == "yes"
        and intervention_auto == "no"
        and cell_line_auto == "no"
        and participant_auto == "yes"
    )
    if hard_exclusion:
        strict_status = "exclude"
    elif ready and (repository != "mbhd" or evidence.mbhd_human_case_control):
        strict_status = "include"
        reasons.append("explicit_human_observational_disease_evidence")
    else:
        strict_status = "needs_manual_review"
        if repository == "mbhd" and not evidence.mbhd_human_case_control:
            reasons.append("microbiomehd_case_control_evidence_not_verified")
        else:
            reasons.append("one_or_more_required_elements_uncertain")

    if strict_status not in STATUS_VALUES:  # defensive invariant
        raise AssertionError(f"invalid strict status: {strict_status}")

    return {
        "species_evidence": " | ".join(species),
        "species_evidence_source": " | ".join(_unique(evidence.species_sources)),
        "human_status": human,
        "disease_like_auto": disease,
        "intervention_exclusion_auto": intervention_auto,
        "cell_line_exclusion_auto": cell_line_auto,
        "human_participant_auto": participant_auto,
        "human_participant_manual": "",
        "manual_disease_like": "",
        "manual_intervention_exclusion": "",
        "manual_final_status": "",
        "strict_observational_status": strict_status,
        "automatic_decision_reason": " | ".join(_unique(reasons)),
        "evidence_source": " | ".join(_unique(evidence.sources)),
        "evidence_text": _clean(evidence_text, limit=4_000),
        "evidence_url": evidence.evidence_url,
        "reviewer_1": "",
        "reviewer_1_decision": "",
        "reviewer_1_notes": "",
        "reviewer_2": "",
        "reviewer_2_decision": "",
        "reviewer_2_notes": "",
        "adjudicator": "",
        "adjudication_decision": "",
        "adjudication_notes": "",
        "adjudication_date": "",
    }


def _load_json(path: Path, fallback: object) -> object:
    if not path.is_file():
        return fallback
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return fallback


def _catalog_map(payload: object, list_key: str, id_key: str) -> dict[str, dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get(list_key), list):
        return {}
    return {str(item[id_key]): item for item in payload[list_key] if isinstance(item, dict) and item.get(id_key)}


class LocalMetadataStore:
    """Read-only index over the catalogs and caches already in the project."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = root / "cache"
        self.cache_mbhd = root / "cache_mbhd"
        self.gxa = _catalog_map(_load_json(root / "gxa_catalog.json", {}), "experiments", "accession")
        self.mw = _catalog_map(_load_json(root / "mw_catalog.json", {}), "studies", "study_id")
        audit_payload = _load_json(root / "audit_texts.json", [])
        self.audit = (
            {str(item["id"]): item for item in audit_payload if isinstance(item, dict) and item.get("id")}
            if isinstance(audit_payload, list)
            else {}
        )

    def collect(self, record: Mapping[str, str]) -> EvidenceBundle:
        repository = _repository(record)
        accession = _clean(record.get("id", ""))
        base_species = _unique((record.get("species", ""),))
        base_species_sources = ("all_datasets_final.csv:species",) if base_species else ()
        base_fragments = _unique(
            (
                f"phenotype={_clean(record.get('phenotype', ''))}",
                f"levels={_clean(record.get('levels', ''))}",
            )
        )
        base = EvidenceBundle(
            species_values=base_species,
            species_sources=base_species_sources,
            sources=("all_datasets_final.csv",),
            fragments=base_fragments,
            evidence_url=_source_url(repository, accession),
        )
        reader = {
            "gxa": self._gxa_evidence,
            "mtbls": self._mtbls_evidence,
            "mw": self._mw_evidence,
            "mbhd": self._mbhd_evidence,
        }.get(repository)
        repository_evidence = reader(accession, record) if reader else EvidenceBundle()
        audit = self.audit.get(accession, {})
        audit_title = _clean(audit.get("title", "")) if isinstance(audit, dict) else ""
        audit_source = ("audit_texts.json:title",) if audit_title else ()
        return _merge_evidence(
            base,
            repository_evidence,
            EvidenceBundle(
                sources=audit_source,
                fragments=(f"publication title={audit_title}",) if audit_title else (),
            ),
        )

    def _gxa_evidence(self, accession: str, record: Mapping[str, str]) -> EvidenceBundle:
        catalog = self.gxa.get(accession, {})
        species = _unique((catalog.get("species", ""),))
        sources = ["gxa_catalog.json"] if catalog else []
        species_sources = ["gxa_catalog.json:species"] if species else []
        fragments: list[str] = []
        if catalog:
            fragments.extend(
                (
                    f"experiment type={_clean(catalog.get('type', ''))}",
                    f"catalog factors={_clean(', '.join(map(str, catalog.get('factors', []))))}",
                )
            )

        sdrf_path = self.cache / f"gxa_sdrf_{accession}.tsv"
        fields: dict[str, set[str]] = {}
        if sdrf_path.is_file():
            sources.append(f"cache/{sdrf_path.name}")
            try:
                with sdrf_path.open(encoding="utf-8", errors="replace", newline="") as handle:
                    for row in csv.reader(handle, delimiter="\t"):
                        if len(row) < 6:
                            continue
                        kind, name, value = (
                            _clean(row[3]),
                            _clean(row[4]),
                            _clean(row[5]),
                        )
                        key = f"{kind.lower()}:{name.lower()}"
                        if name.lower() == "organism" and value:
                            species += (value,)
                            species_sources.append(f"cache/{sdrf_path.name}:organism")
                        if _relevant_metadata_name(name):
                            fields.setdefault(key, set()).add(value)
            except OSError:
                fields = {}
        fragments.extend(_condense_fields(fields))
        joined = " ; ".join(fragments)
        factor_names = " ".join(str(value) for value in catalog.get("factors", []))
        explicit_cell_line = bool(any("cell line" in key for key in fields) or _CELL_LINE.search(joined))
        explicit_participant = (
            bool(
                any(re.search(r"individual|subject|patient|donor", key, re.I) for key in fields)
                or _BIOSPECIMEN.search(joined)
                or _PARTICIPANT.search(joined)
            )
            and not explicit_cell_line
        )
        explicit_intervention = bool(
            _INTERVENTION.search(factor_names)
            or any(_INTERVENTION.search(key) for key in fields)
            or _INTERVENTION.search(joined)
            or _INTERVENTION.search(_clean(record.get("phenotype", "")))
        )
        return EvidenceBundle(
            species_values=_unique(species),
            species_sources=_unique(species_sources),
            sources=_unique(sources),
            fragments=_unique(fragments),
            explicit_participant=explicit_participant,
            explicit_cell_line=explicit_cell_line,
            explicit_case_control=_case_control_from_record(record),
            explicit_intervention=explicit_intervention,
        )

    def _mtbls_evidence(self, accession: str, record: Mapping[str, str]) -> EvidenceBundle:
        sources: list[str] = []
        fragments: list[str] = []
        species: list[str] = []
        species_sources: list[str] = []

        sample_path = self.cache / f"f_{accession}_s_{accession}.txt"
        fields: dict[str, set[str]] = {}
        if sample_path.is_file():
            sources.append(f"cache/{sample_path.name}")
            try:
                with sample_path.open(encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle, delimiter="\t")
                    header = next(reader, [])
                    relevant = [i for i, name in enumerate(header) if _relevant_metadata_name(name)]
                    organism = [
                        i for i, name in enumerate(header) if re.search(r"characteristics\[organism\]$", name, re.I)
                    ]
                    for row in reader:
                        for index in organism:
                            if index < len(row) and _clean(row[index]):
                                species.append(_clean(row[index]))
                                species_sources.append(f"cache/{sample_path.name}:Characteristics[Organism]")
                        for index in relevant:
                            if index < len(row) and _clean(row[index]):
                                fields.setdefault(_clean(header[index]).lower(), set()).add(_clean(row[index]))
            except OSError:
                fields = {}
        sample_fragments = _condense_fields(fields)
        fragments.extend(sample_fragments)

        inv_path = self.cache / f"inv_{accession}.txt"
        design_text = ""
        if inv_path.is_file():
            sources.append(f"cache/{inv_path.name}")
            selected: list[str] = []
            try:
                with inv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
                    for row in csv.reader(handle, delimiter="\t"):
                        if not row:
                            continue
                        key = _clean(row[0])
                        if key in {
                            "Investigation Title",
                            "Study Title",
                            "Study Description",
                            "Study Design Type",
                        }:
                            value = _clean(" | ".join(part for part in row[1:] if _clean(part)))
                            if value:
                                selected.append(f"{key}={value}")
                                if key in {
                                    "Investigation Title",
                                    "Study Title",
                                    "Study Design Type",
                                }:
                                    design_text += f" {value}"
            except OSError:
                selected = []
            fragments.extend(selected)

        joined = " ; ".join(fragments)
        explicit_cell_line = bool(any("cell line" in key for key in fields) or _CELL_LINE.search(design_text))
        explicit_participant = (
            bool(
                any(re.search(r"individual|subject|patient|donor", key, re.I) for key in fields)
                or _PARTICIPANT.search(joined)
            )
            and not explicit_cell_line
        )
        explicit_intervention = bool(
            any(_INTERVENTION.search(key) for key in fields)
            or _INTERVENTION.search(" ; ".join(sample_fragments))
            or _INTERVENTION.search(_clean(record.get("phenotype", "")))
            or _INTERVENTION.search(design_text)
        )
        explicit_case_control = bool(_case_control_from_record(record) or _OBSERVATIONAL.search(design_text))
        return EvidenceBundle(
            species_values=_unique(species),
            species_sources=_unique(species_sources),
            sources=_unique(sources),
            fragments=_unique(fragments),
            explicit_participant=explicit_participant,
            explicit_cell_line=explicit_cell_line,
            explicit_case_control=explicit_case_control,
            explicit_intervention=explicit_intervention,
        )

    def _mw_evidence(self, accession: str, record: Mapping[str, str]) -> EvidenceBundle:
        catalog = self.mw.get(accession, {})
        species = _unique((catalog.get("species", ""),))
        sources = ["mw_catalog.json"] if catalog else []
        species_sources = ["mw_catalog.json:species"] if species else []
        title = _clean(catalog.get("study_title", ""))
        fragments = [f"study title={title}"] if title else []

        factors_path = self.cache / f"mw_fac_{accession}.json"
        factor_fields: dict[str, set[str]] = {}
        sample_sources: set[str] = set()
        if factors_path.is_file():
            payload = _load_json(factors_path, {})
            if isinstance(payload, dict):
                sources.append(f"cache/{factors_path.name}")
                for item in payload.values():
                    if not isinstance(item, dict):
                        continue
                    sample_source = _clean(item.get("sample_source", ""))
                    if sample_source:
                        sample_sources.add(sample_source)
                    for part in str(item.get("factors", "")).split("|"):
                        if ":" not in part:
                            continue
                        key, value = map(_clean, part.split(":", 1))
                        if key:
                            factor_fields.setdefault(key.lower(), set()).add(value)
        if sample_sources:
            fragments.append(f"sample source={', '.join(sorted(sample_sources)[:8])}")
        fragments.extend(_condense_fields(factor_fields))
        joined = " ; ".join(fragments)
        explicit_cell_line = bool(
            _CELL_LINE.search(title)
            or any("cell line" in key for key in factor_fields)
            or _CELL_LINE.search(" ".join(sample_sources))
        )
        explicit_participant = (
            bool(
                any(re.search(r"subject|patient|participant|donor", key, re.I) for key in factor_fields)
                or _PARTICIPANT.search(joined)
            )
            and not explicit_cell_line
        )
        explicit_intervention = bool(
            any(_INTERVENTION.search(key) for key in factor_fields)
            or _INTERVENTION.search(joined)
            or _INTERVENTION.search(_clean(record.get("phenotype", "")))
            or _INTERVENTION.search(title)
        )
        return EvidenceBundle(
            species_values=species,
            species_sources=_unique(species_sources),
            sources=_unique(sources),
            fragments=_unique(fragments),
            explicit_participant=explicit_participant,
            explicit_cell_line=explicit_cell_line,
            explicit_case_control=bool(_case_control_from_record(record) or _OBSERVATIONAL.search(title)),
            explicit_intervention=explicit_intervention,
        )

    def _mbhd_evidence(self, accession: str, record: Mapping[str, str]) -> EvidenceBundle:
        archive = self.cache_mbhd / f"{accession}_results.tar.gz"
        if not archive.is_file():
            return EvidenceBundle()
        species: list[str] = []
        states: list[str] = []
        sample_types: list[str] = []
        try:
            with tarfile.open(archive, "r:gz") as tar:
                member = next(
                    (
                        item
                        for item in tar.getmembers()
                        if item.isfile()
                        and item.name.endswith(".metadata.txt")
                        and not item.name.rsplit("/", 1)[-1].startswith("._")
                    ),
                    None,
                )
                extracted = tar.extractfile(member) if member else None
                raw = extracted.read() if extracted else b""
            decoded = raw.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")
            for row in reader:
                species.append(_clean(row.get("Organism", "")))
                states.append(_clean(row.get("DiseaseState", row.get("Disease_State", ""))))
                sample_types.append(_clean(row.get("SampleType", "")))
        except (OSError, tarfile.TarError):
            return EvidenceBundle(sources=(f"cache_mbhd/{archive.name}:unreadable",))

        species_values = _unique(species)
        state_values = _unique(states)
        sample_type_values = _unique(sample_types)
        all_human = _human_status(species_values) == "human"
        healthy = any(_CONTROL.match(state) for state in state_values)
        cases = any(not _CONTROL.match(state) for state in state_values if state)
        mbhd_supported = bool(all_human and healthy and cases)
        fragments = _unique(
            (
                f"Organism={', '.join(species_values)}",
                f"SampleType={', '.join(sample_type_values)}",
                f"DiseaseState={', '.join(state_values)}",
                f"mbhd_human_case_control_supported={str(mbhd_supported).lower()}",
            )
        )
        return EvidenceBundle(
            species_values=species_values,
            species_sources=(f"cache_mbhd/{archive.name}:Organism",),
            sources=(f"cache_mbhd/{archive.name}",),
            fragments=fragments,
            explicit_participant=mbhd_supported,
            explicit_case_control=mbhd_supported,
            mbhd_human_case_control=mbhd_supported,
        )


def _relevant_metadata_name(name: str) -> bool:
    return bool(
        re.search(
            r"organism$|organism part|sample type|disease|diagnos|condition|"
            r"phenotype|group|cell line|individual|subject|patient|donor|"
            r"treatment|compound|dose|drug|intervention|stimulus|exposure",
            _clean(name),
            re.I,
        )
    )


def _condense_fields(fields: Mapping[str, set[str]]) -> tuple[str, ...]:
    fragments: list[str] = []
    for key in sorted(fields):
        values = sorted(_unique(fields[key]))
        if re.search(r"individual|subject|patient|donor", key, re.I):
            fragments.append(f"{key}=present")
        elif values:
            shown = values[:6]
            suffix = f" (+{len(values) - len(shown)} more)" if len(values) > len(shown) else ""
            fragments.append(f"{key}={', '.join(shown)}{suffix}")
    return _unique(fragments)


def _merge_evidence(*items: EvidenceBundle) -> EvidenceBundle:
    return EvidenceBundle(
        species_values=_unique(value for item in items for value in item.species_values),
        species_sources=_unique(value for item in items for value in item.species_sources),
        sources=_unique(value for item in items for value in item.sources),
        fragments=_unique(value for item in items for value in item.fragments),
        evidence_url=next((item.evidence_url for item in items if item.evidence_url), ""),
        explicit_participant=any(item.explicit_participant for item in items),
        explicit_cell_line=any(item.explicit_cell_line for item in items),
        explicit_case_control=any(item.explicit_case_control for item in items),
        explicit_intervention=any(item.explicit_intervention for item in items),
        mbhd_human_case_control=any(item.mbhd_human_case_control for item in items),
    )


def build_cohort_rows(input_csv: str | Path, metadata_root: str | Path | None = None) -> list[dict[str, str]]:
    """Build one review-template row per unique legacy dataset row."""

    source = Path(input_csv)
    root = Path(metadata_root) if metadata_root is not None else source.parent
    store = LocalMetadataStore(root)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("input CSV must contain an 'id' column")
        input_rows = list(reader)
    identifiers = [_clean(row.get("id", "")) for row in input_rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every input row must have a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("input dataset ids must be unique")

    output: list[dict[str, str]] = []
    for record in input_rows:
        evidence = store.collect(record)
        automatic = classify_record(record, evidence)
        base = {
            "id": _clean(record.get("id", "")),
            "repository": _repository(record),
            "platform": _clean(record.get("plat") or record.get("platform", "")),
            "n_samples": _clean(record.get("n_samples", "")),
            "n_features": _clean(record.get("n_features", "")),
            "n_pos": _clean(record.get("n_pos", "")),
            "n_neg": _clean(record.get("n_neg", "")),
            "phenotype": _clean(record.get("phenotype", "")),
            "levels": _clean(record.get("levels", "")),
        }
        output.append({field: {**base, **automatic}.get(field, "") for field in OUTPUT_FIELDS})
    return output


def write_cohort_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    metadata_root: str | Path | None = None,
) -> int:
    """Write a new scaffold CSV, refusing input mutation or output overwrite."""

    source = Path(input_csv)
    destination = Path(output_csv)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output paths must be different")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    rows = build_cohort_rows(source, metadata_root=metadata_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="legacy measurements CSV")
    parser.add_argument("--output", required=True, type=Path, help="new review-template CSV")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="directory containing local catalogs/cache (default: input directory)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="optional row-count assertion (use 1102 for the full benchmark)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = build_cohort_rows(args.input, metadata_root=args.metadata_root)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {len(rows)}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output paths must be different")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to new file: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
