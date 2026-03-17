# CLAUDE.md

This file provides the repo-specific guidance that is most useful to Claude Code.
Human-facing setup, architecture, and scientific background live in `README.md`.

## Core Constraints

- **Python 3.11 only** (pinned in `.python-version`).
- Use **`uv`** for dependency management and execution.
- See `README.md` for setup, smoke tests, architecture, memory-likelihood math,
  and waveform implementation notes.
- The `memory` package is installed in dev mode via `uv sync`.

## Operational Rules

- **Never run production analyses locally.** Submit them via `submit.sh`.
- Production task commands must set `TGRPOP_DEVICE_COUNT=4` to use all allocated GPUs.
- The `data/NRSur7dq4.h5` symlink must exist for surrogate evaluation to work.
- JAX platform/device setup in `scripts/run_hierarchical_analysis.py` happens at
  module import time and must remain **before** other imports that initialize JAX.

## Scientific Calibration

- `memory/hierarchical/models.py` is calibrated to the GWTC-4 populations paper
  (Appendix B.3 / Table 6). Treat changes to the `PRIOR` dict or `mmin = 3` as
  scientific-model changes, not refactors.

## Waveform Invariants

- `compute_one_sample_fd()` and `evaluate_surrogate_with_LAL()` must use the
  same effective per-sample `minimum_frequency`.
- ISCO-limit failures may only appear on C-level stderr, so retry logic must
  inspect both the Python exception text and captured stderr.
- On the **first** ISCO failure, do not apply the no-progress guard unless
  `waveform_arguments["minimum_frequency"]` was already explicitly lowered by a
  prior retry.
- After retrying, write the resolved `minimum_frequency` back to the sample so
  the memory path reuses it; reset shared waveform-generator state before the
  next sample.
- For `pyseobnr` Nyquist failures, retry with `lmax_nyquist=2` and then `1`.
- Extra PE `waveform_arguments_dict` flags must propagate to both the bilby and
  direct-LAL waveform paths.

## Common Code Patterns

- HDF5 parameter lookup uses multiple naming conventions with fallbacks.
- bilby config parsing also handles several alias patterns for the same setting.
