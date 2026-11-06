"""Wall-clock time of presplit-audit on simulated signal-free data of typical sizes.

Sizes span the corpus: its median dataset (56 samples, 1,875 features) and larger cases. Each audit
runs both protocols with and without 199 permutations, on 8 worker processes. Hardware and software
are recorded with the timings.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool" / "src"))
from presplit_audit import audit  # noqa: E402

SIZES = [(40, 1000), (56, 1875), (100, 5000), (300, 20000)]
STEPS, JOBS, B = ("top_q", "pls_da"), 8, 199


def cpu() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                   # recorded, not fatal
        return platform.processor()


def main() -> None:
    rows = []
    for n, p in SIZES:
        rng = np.random.default_rng(n * 100003 + p)
        X, y = rng.normal(size=(n, p)), np.repeat([0, 1], n // 2)
        for step in STEPS:
            for b in (0, B):
                t0 = time.perf_counter()
                audit(X, y, step, n_permutations=b, random_state=0, n_jobs=JOBS)
                rows.append({"n": n, "p": p, "step": step, "permutations": b,
                             "seconds": round(time.perf_counter() - t0, 2)})
                print(rows[-1], flush=True)
    out = {"cpu": cpu(), "workers": JOBS, "python": platform.python_version(), "rows": rows}
    (HERE / "results" / "runtime_benchmark.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
