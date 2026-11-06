"""Audit a simulated dataset with no signal, then one with a weak signal.

Run from the tool folder: ``python examples/example_synthetic.py``.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from presplit_audit import audit  # noqa: E402

rng = np.random.default_rng(7)
y = np.repeat([0, 1], 20)
noise = rng.normal(size=(40, 1000))
weak = noise.copy()
weak[:, :10] += 0.6 * y[:, None]

for name, X in (("no signal", noise), ("weak signal in 10 of 1,000 features", weak)):
    print(f"--- {name}")
    print(audit(X, y, "top_q", n_permutations=99, random_state=0, n_jobs=4).summary())
