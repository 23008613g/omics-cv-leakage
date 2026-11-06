"""Regenerate included summaries in a temporary directory and compare values."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = Path("manuscript_B/patterns_submission")
JOBS = [
    ("validation/summarize_validation.py", "validation/results/validation_summary.json"),
    ("analyses/extra_analyses.py", "analyses/results/extra_results.json"),
    ("analyses/metrics_lasso_summary.py", "analyses/results/metrics_lasso_summary.json"),
    ("analyses/duplicate_summary.py", "analyses/results/duplicate_summary.json"),
    ("analyses/cv_folds_summary.py", "analyses/results/cv_folds_summary.json"),
    ("rereview/compare_with_original.py", "rereview/agreement_summary.json"),
    ("rereview/subset_sensitivity.py", "rereview/subset_sensitivity.json"),
    ("reserve/summarize_reserve.py", "reserve/results/cross_cohort_summary.json"),
    ("analyses/cross_study_gxa_summary.py", "analyses/results/cross_study_gxa_summary.json"),
]

def compare(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: keys changed")
        for k in a:
            compare(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: lengths changed")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not ((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)):
            raise AssertionError(f"{path}: {a} != {b}")
    elif a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-full-checkpoints", action="store_true")
    args = ap.parse_args()
    jobs = list(JOBS)
    if args.with_full_checkpoints:
        if not os.environ.get("PRESPLIT_FORMAL_DEPOSIT"):
            raise RuntimeError("Set PRESPLIT_FORMAL_DEPOSIT to the extracted release asset first")
        jobs.append(("analyses/cv_folds_summary.py --all", "analyses/results/cv_folds_all_summary.json"))
    with tempfile.TemporaryDirectory(prefix="omics-summary-check-") as folder:
        work = Path(folder) / "repository"
        shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache"))
        for script, result in jobs:
            name, *arguments = script.split()
            run = subprocess.run([sys.executable, str(work / BASE / name), *arguments], cwd=work,
                                 text=True, capture_output=True, timeout=600)
            if run.returncode:
                raise RuntimeError(f"{script}: {run.stderr}")
            compare(json.loads((ROOT / BASE / result).read_text()), json.loads((work / BASE / result).read_text()))
            print(f"PASS {script}", flush=True)
    print(f"PASS {len(jobs)} numerical summaries; archived records unchanged", flush=True)

if __name__ == "__main__":
    main()
