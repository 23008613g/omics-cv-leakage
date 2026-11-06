"""Check public paths and distinguish released sources from historical hashes."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "mtbls_screen"
CONTROL = ROOT / "manuscript_B/benchmark_analyses/control_arm"
ANALYSES = ROOT / "manuscript_B/patterns_submission/analyses"
MANIFEST = RUNNER / "matched_supervised_v4/formal_run_v4/manifest.json"


def control_module():
    sys.path.insert(0, str(CONTROL))
    return importlib.import_module("run_control_arm")


def test_public_directories_exclude_historical_submission_names():
    assert (RUNNER / "benchmark_v4").is_dir()
    assert CONTROL.is_dir()
    assert not (RUNNER / "nm_v4").exists()
    assert not (ROOT / "manuscript_B/nm_brief_communication").exists()


def test_historical_hashes_are_preserved_and_release_sources_are_separate():
    manifest = json.loads(MANIFEST.read_text())
    historical = manifest["code_file_sha256"]
    encoded = json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "b2f7dde3b3b10b4a32bd6edc496f00dd096d502b7467474f64430b6ca5f4908d"
    mapping = manifest["historical_to_release_paths"]
    current = manifest["release_code_file_sha256"]
    assert set(mapping) == set(historical)
    assert set(mapping.values()) == set(current)
    assert len(current) == 18
    for path, digest in current.items():
        assert hashlib.sha256((RUNNER / path).read_bytes()).hexdigest() == digest
    report = control_module().verify_formal_code()
    assert set(report) == set(current)
    assert all(report.values())


def test_release_source_check_detects_modified_file(tmp_path, monkeypatch):
    module = control_module()
    manifest = json.loads(MANIFEST.read_text())
    for path in manifest["release_code_file_sha256"]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RUNNER / path, target)
    target = "benchmark_v4/protocol.py"
    with (tmp_path / target).open("a") as handle:
        handle.write("\n# deliberate integrity-test change\n")
    monkeypatch.setattr(module, "RUNNER_ROOT", tmp_path)
    report = module.verify_formal_code()
    assert not report[target]
    assert sum(not ok for ok in report.values()) == 1


def test_package_binding_accepts_duplicate_local_paths_but_rejects_foreign_sources(tmp_path, monkeypatch):
    module = control_module()
    local = str(RUNNER / "benchmark_v4")
    monkeypatch.setitem(sys.modules, "benchmark_v4", SimpleNamespace(__path__=[local, local]))
    module._bind_formal_package()
    monkeypatch.setitem(sys.modules, "benchmark_v4", SimpleNamespace(__path__=[str(tmp_path)]))
    with pytest.raises(RuntimeError, match="different benchmark_v4"):
        module._bind_formal_package()


def test_default_whitelist_uses_public_records_with_locked_canonical_values():
    sys.path.insert(0, str(RUNNER))
    runner = importlib.import_module("benchmark_v4.run_benchmark")
    assert runner.DEFAULT_WHITELIST.is_file()
    records = runner.load_whitelist()
    assert len(records) == 1102
    assert len({row.dataset_id for row in records}) == 1102
    fingerprint, files = runner.code_fingerprint()
    assert fingerprint
    assert len(files) == 18
    assert all((RUNNER / path).is_file() for path in files)


def test_ten_fold_analysis_uses_current_public_per_dataset_table():
    sys.path.insert(0, str(ANALYSES))
    module = importlib.import_module("cv_folds_all")
    assert module.DATA_S1 == ROOT / "manuscript_B/patterns_submission/output/Patterns_Data_S1_per_dataset.csv"
    data = module.analysed()
    assert len(data) == 1086
    assert data.dataset_id.nunique() == 1086
    assert data.top_k_pre_split_auc.notna().all()
