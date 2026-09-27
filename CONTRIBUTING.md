# Contributing

Changes should keep the correspondence between the manuscript and the implementation explicit. A bug fix should identify the incorrect behavior; a new method variant should be named and documented as a variant.

## Local setup

Use a virtual environment with Python 3.10 or later. Install a suitable PyTorch build, then run:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
python examples/smoke_train.py
python -m build
```

## A useful change report

Describe the problem, implementation change, and verification. For numerical changes, include the relevant input shape, dtype, device, mode, seed, and PyTorch version. Add a focused regression test where it protects a meaningful property such as initial output equivalence, coordinate mapping, checkpoint loading, or gradient flow.

Public API changes should update the English and Chinese READMEs and the affected method/integration documentation. Preserve the default camera-then-LiDAR input convention. Do not silently change feature-grid geometry or reinterpret residual weights as sensor confidence.

## Experimental claims

A synthetic loss decrease is not a detector result. New benchmark claims should include complete configurations, the upstream detector commit, checkpoint identifiers/checksums, dataset and split information, evaluation commands, and logs. Label the results as produced by this reimplementation unless their original provenance is established.

Do not commit datasets, large checkpoints, access tokens, or private paths. Check `git status` and the staged diff before submission. Contributions to this implementation use the project's MIT License. External code must retain its provenance and satisfy its own license.
