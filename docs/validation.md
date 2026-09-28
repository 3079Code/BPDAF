# Validation record

Local checks for this implementation were completed on 2026-09-27 with
Python 3.10.11 and PyTorch 2.11.0+cpu on Windows. No CUDA device was available.

| Check | Result |
| --- | --- |
| Unit and contract tests | 34 passed |
| Ruff lint and formatting | Passed |
| Synthetic optimizer example | Five finite optimizer steps; initial reference output exactly equal |
| Synthetic checkpoint round trip | Reloaded output exactly equal |
| Source and wheel distributions | Both built successfully |

## Maintainer-reported dataset reproduction

The repository maintainer reports that nuScenes and KITTI reproduction has
been completed and that the results match the manuscript. This status is
based on the maintainer's confirmation, not on the CPU checks above. The
corresponding run logs, checkpoints, full configurations, and tested commit
mapping are not included in this release, so this record does not constitute
an independent verification of the dataset results.

The final numerical tables reported in the manuscript are collected in
[results.md](results.md). They are provided as a compact result record rather
than as raw training or evaluation logs.

## Running the local checks

Reproduce the local checks from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python examples/smoke_train.py --steps 5 --output runs/smoke.pt
python -m build
```

Tests cover cell-center geometry on rectangular and displaced grids, Fourier
channel order, per-sample energy normalization, zero features, cue detachment,
complementary gates, train/eval reference equivalence, branch RNG conservation,
the staged gradient path, reference checkpoint loading, full state round trips,
CPU bfloat16 autocast, and the optional adapter's coordinate convention.

Adapter tests use a mocked MMDetection3D registry. They validate registration
and tensor/coordinate contracts without claiming that CUDA extensions, dataset
pipelines or a full detector have run. Check the actual training environment
before making detector-level compatibility or performance claims.

GitHub Actions additionally runs Linux CPU checks for Python 3.10 with
PyTorch 2.1.2 and Python 3.12 with PyTorch 2.6.0. Both environments passed for
commit `52ff7ca` in [the initial run](https://github.com/3079Code/BPDAF/actions/runs/36331466622),
including lint, formatting, tests, the synthetic example, and package building.
Results for subsequent commits are visible in
[the workflow runs](https://github.com/3079Code/BPDAF/actions/workflows/ci.yml).
These CPU checks do not establish full-dataset reproduction.
