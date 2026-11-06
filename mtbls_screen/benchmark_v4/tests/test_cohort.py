from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from cohort import (  # noqa: E402
    EvidenceBundle,
    build_cohort_rows,
    classify_record,
    write_cohort_csv,
)


def _row(**updates: str) -> dict[str, str]:
    row = {
        "id": "TEST-1",
        "plat": "transcriptomics",
        "platform": "transcriptomics",
        "repository": "",
        "phenotype": "disease",
        "levels": "['type 2 diabetes', 'healthy control']",
        "species": "Homo sapiens",
    }
    return {**row, **updates}


def _evidence(
    *fragments: str,
    species: tuple[str, ...] = ("Homo sapiens",),
    participant: bool = False,
    cell_line: bool = False,
    case_control: bool = False,
    intervention: bool = False,
    mbhd_case_control: bool = False,
) -> EvidenceBundle:
    return EvidenceBundle(
        species_values=species,
        sources=("test fixture",),
        fragments=fragments,
        evidence_url="https://example.org/TEST-1",
        explicit_participant=participant,
        explicit_cell_line=cell_line,
        explicit_case_control=case_control,
        explicit_intervention=intervention,
        mbhd_human_case_control=mbhd_case_control,
    )


def test_clear_human_disease_case_control_is_included() -> None:
    evidence = _evidence(
        "case-control cohort of patients with type 2 diabetes and healthy controls",
        "sample type=serum; characteristic individual present",
        participant=True,
        case_control=True,
    )

    result = classify_record(_row(), evidence)

    assert result["human_status"] == "human"
    assert result["disease_like_auto"] == "yes"
    assert result["intervention_exclusion_auto"] == "no"
    assert result["human_participant_auto"] == "yes"
    assert result["strict_observational_status"] == "include"


def test_human_cell_line_is_never_promoted_to_strict_include() -> None:
    evidence = _evidence(
        "characteristic cell line=MCF-7; factor disease=breast carcinoma",
        cell_line=True,
        case_control=True,
    )

    result = classify_record(_row(), evidence)

    assert result["cell_line_exclusion_auto"] == "yes"
    assert result["human_participant_auto"] == "no"
    assert result["strict_observational_status"] == "exclude"


def test_treatment_contrast_is_excluded_even_with_human_participants() -> None:
    evidence = _evidence(
        "randomized intervention in human volunteers",
        participant=True,
        intervention=True,
    )
    row = _row(phenotype="Treatment", levels="['drug', 'placebo']")

    result = classify_record(row, evidence)

    assert result["intervention_exclusion_auto"] == "yes"
    assert result["strict_observational_status"] == "exclude"


def test_nonhuman_dataset_is_excluded() -> None:
    evidence = _evidence(
        "mouse disease model and wild-type controls",
        species=("Mus musculus",),
        participant=True,
        case_control=True,
    )

    result = classify_record(_row(species="Mus musculus"), evidence)

    assert result["human_status"] == "nonhuman"
    assert result["strict_observational_status"] == "exclude"


def test_clear_non_disease_factor_is_not_relabelled_by_control_word() -> None:
    evidence = _evidence(
        "human participant case-control sampling",
        participant=True,
        case_control=True,
    )
    row = _row(phenotype="genotype", levels="['case', 'control']")

    result = classify_record(row, evidence)

    assert result["disease_like_auto"] == "no"
    assert result["strict_observational_status"] == "exclude"


def test_uncertain_species_or_participant_status_requires_manual_review() -> None:
    evidence = _evidence(
        "condition A versus condition B",
        species=(),
    )

    result = classify_record(
        _row(species="", phenotype="Condition", levels="['A', 'B']"),
        evidence,
    )

    assert result["human_status"] == "uncertain"
    assert result["human_participant_auto"] == "uncertain"
    assert result["strict_observational_status"] == "needs_manual_review"


def test_disease_label_without_case_control_evidence_requires_manual_review() -> None:
    evidence = _evidence(
        "human participant biospecimen",
        participant=True,
        case_control=False,
    )

    result = classify_record(
        _row(phenotype="DiseaseState", levels="['NAFLD']"),
        evidence,
    )

    assert result["strict_observational_status"] == "needs_manual_review"


def test_nondisease_microbiome_case_control_label_is_not_auto_included() -> None:
    row = _row(
        id="weight_example",
        plat="microbiome",
        platform="microbiome",
        phenotype="DiseaseState",
        levels="['overweight', 'healthy control']",
        species="",
    )
    evidence = _evidence(
        "Organism=human; SampleType=stool; DiseaseState=overweight,H",
        participant=True,
        case_control=True,
        mbhd_case_control=True,
    )

    result = classify_record(row, evidence)

    assert result["disease_like_auto"] != "yes"
    assert result["strict_observational_status"] == "needs_manual_review"


def test_microbiomehd_requires_explicit_human_case_control_evidence() -> None:
    row = _row(
        id="crc_zeller",
        plat="microbiome",
        platform="microbiome",
        phenotype="DiseaseState",
        levels="['CRC', 'H']",
        species="",
    )
    unsupported = _evidence(
        "Organism=human; DiseaseState=CRC,H",
        participant=True,
        case_control=True,
    )
    supported = _evidence(
        "Organism=human; SampleType=stool; DiseaseState=CRC,H",
        participant=True,
        case_control=True,
        mbhd_case_control=True,
    )

    result_unsupported = classify_record(row, unsupported)
    result_supported = classify_record(row, supported)

    assert result_unsupported["strict_observational_status"] == "needs_manual_review"
    assert result_supported["strict_observational_status"] == "include"


def test_gxa_stimulus_metadata_prevents_automatic_observational_include(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "gxa_catalog.json").write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "accession": "E-TEST-1",
                        "species": "Homo sapiens",
                        "type": "RNASEQ_MRNA_DIFFERENTIAL",
                        "factors": ["disease"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (cache / "gxa_sdrf_E-TEST-1.tsv").write_text(
        "E-TEST-1\t\ts1\tcharacteristic\torganism\tHomo sapiens\n"
        "E-TEST-1\t\ts1\tcharacteristic\tindividual\tP01\n"
        "E-TEST-1\t\ts1\tcharacteristic\tcondition\tphysical exercise\n"
        "E-TEST-1\t\ts1\tfactor\tdisease\tcase\n",
        encoding="utf-8",
    )
    input_csv = tmp_path / "legacy.csv"
    row = _row(id="E-TEST-1", species="")
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    [result] = build_cohort_rows(input_csv, metadata_root=tmp_path)

    assert result["intervention_exclusion_auto"] == "yes"
    assert result["strict_observational_status"] == "exclude"
    assert "P01" not in result["evidence_text"]
    assert "individual=present" in result["evidence_text"]


def test_build_preserves_row_count_and_leaves_review_fields_blank(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "legacy.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_row()))
        writer.writeheader()
        writer.writerow(_row())
        writer.writerow(
            _row(
                id="TEST-2",
                species="Mus musculus",
                phenotype="genotype",
                levels="['mutant', 'wild type']",
            )
        )

    rows = build_cohort_rows(input_csv, metadata_root=tmp_path)

    assert [row["id"] for row in rows] == ["TEST-1", "TEST-2"]
    assert all(row["strict_observational_status"] in {"include", "exclude", "needs_manual_review"} for row in rows)
    for row in rows:
        assert row["reviewer_1"] == ""
        assert row["reviewer_2"] == ""
        assert row["adjudication_decision"] == ""
        assert row["human_participant_manual"] == ""


def test_writer_creates_new_csv_without_mutating_or_overwriting(tmp_path: Path) -> None:
    input_csv = tmp_path / "legacy.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_row()))
        writer.writeheader()
        writer.writerow(_row())
    before = hashlib.sha256(input_csv.read_bytes()).hexdigest()

    output_csv = tmp_path / "cohort_review.csv"
    count = write_cohort_csv(input_csv, output_csv, metadata_root=tmp_path)

    assert count == 1
    assert output_csv.exists()
    assert hashlib.sha256(input_csv.read_bytes()).hexdigest() == before

    with pytest.raises(FileExistsError):
        write_cohort_csv(input_csv, output_csv, metadata_root=tmp_path)
    with pytest.raises(ValueError, match="different"):
        write_cohort_csv(input_csv, input_csv, metadata_root=tmp_path)


def test_full_local_scaffold_has_exactly_one_row_per_benchmark_dataset() -> None:
    metadata_root = MODULE_DIR.parent
    source = metadata_root / "all_datasets_final.csv"

    rows = build_cohort_rows(source, metadata_root=metadata_root)

    assert len(rows) == 1_102
    assert len({row["id"] for row in rows}) == 1_102
    assert all(row["evidence_source"] for row in rows)
    assert all(row["strict_observational_status"] in {"include", "exclude", "needs_manual_review"} for row in rows)
