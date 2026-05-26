# Constraining Gravitational Wave Memory with Hierarchical Inference

This repository contains the analysis code used to produce the results of

> Mitman, Isi, & Farr,
> *Constraining Gravitational Wave Memory with Hierarchical Inference*,
> 
> arXiv:XXXX.XXXXX (LIGO Document [LIGO-P2600229](https://dcc.ligo.org/LIGO-P2600229)).

Data produced by the code in this repository (up to RNG seed values) are available at
[10.5281/zenodo.20347300](https://doi.org/10.5281/zenodo.20347300).

The methodology and the scientific results are described in the paper;
this README only covers what is needed to reproduce them.

## Installation

The environment is specified in [`pyproject.toml`](pyproject.toml) and
can be reproduced with any compatible package manager. For example, using
[`uv`](https://docs.astral.sh/uv/),

```bash
uv sync
```

from the repo root resolves and installs all dependencies, including the
local `memory` package.

A few data assets (the NRSur7dq4 surrogate and the selection-injection
files) are not in the repo. Fetch them with

```bash
cd data && ./download.sh
```

## Input data

### Posterior samples

Per-event parameter estimation samples were downloaded from the public
GWTC releases listed in [`provenance.txt`](provenance.txt):

| Catalog   | Version | Source                                                |
|-----------|---------|-------------------------------------------------------|
| GWTC-2.1  | v2      | [10.5281/zenodo.6513631](https://doi.org/10.5281/zenodo.6513631) |
| GWTC-3    | v2      | [10.5281/zenodo.8177023](https://doi.org/10.5281/zenodo.8177023) |
| GWTC-4    | v2      | [10.5281/zenodo.17014085](https://doi.org/10.5281/zenodo.17014085) |

For GWTC-3 we use the `*_nocosmo*` files; their `log_prior` is consistent with
the sample distribution, whereas the `cosmo` files have been resampled but
retain the original prior field.

### Selection injections

The hierarchical analysis uses the LVK GWTC-4.0 sensitivity-estimate
injections of Essick et al. ([arXiv:2508.10638](https://arxiv.org/abs/2508.10638),
dataset [10.5281/zenodo.16740128](https://doi.org/10.5281/zenodo.16740128)).
The default file used for the paper is

```
data/selection/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf
```

which provides cumulative O1+O2+O3+O4a sensitivity (semi-analytic for O1+O2,
search-pipeline-recovered for O3+O4a). Other selection files used for
robustness checks are downloaded by `data/download.sh`.

## Reproducing the analysis

Four scripts cover the full pipeline. Below we describe their typical usage,
but users will find that certain arguments need to be set/modified based on their setup
(e.g., there are user-dependent paths that need to be specified).

1. [`scripts/compute_gw_memory_for_GWTC.py`](scripts/compute_gw_memory_for_GWTC.py) —
   computes per-event memory-amplitude posteriors $(\mu_{s}, \sigma_{s})$ from
   the GWTC parameter-estimation files. Outputs one
   `{output_dir}/{event_name}/memory_results.h5` file per event as well as
   diagnostic plots for each waveform model.

   ```bash
   uv run python scripts/compute_gw_memory_for_GWTC.py \
       --base-dirs /path/to/GWTC-releases \
       --output-dir results/memory_gwtc/
   ```

2. [`scripts/run_hierarchical_analysis.py`](scripts/run_hierarchical_analysis.py) —
   runs the hierarchical (numpyro / NUTS) inference on top of those per-event
   memory measurements, jointly with the astrophysical population. Three
   analysis modes are available via `--analyze`: `astro` (population only),
   `memory` (TGR-only hyperparameters), and `joint` (both).

   ```bash
   uv run python scripts/run_hierarchical_analysis.py \
       "path/to/posteriors/*.h5" \
       --memory-dir results/memory_gwtc/ \
       --injection-file data/selection/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf \
       --analyze astro memory joint \
       --outdir results/hierarchical/
   ```

3. [`scripts/forecast_A_parametric_hyperposterior.py`](scripts/forecast_A_parametric_hyperposterior.py) —
   runs the forecast analysis on top of the hierarchical inference results; this should be run for both the
   ``joint`` results and the ``memory results`.

   ```bash
   uv run python scripts/forecast_A_parametric_hyperposterior.py \
   --posterior-nc results/hierarchical/result_joint.nc \
   --include-events-file results/hierarchical/analyzed_events.txt \
   --mu-scale-power 0.5 \
   --sigma2-scale-power 1.0 \
   --sigma2-posterior invchi2 \
   --outdir results/forecast/forecast_parametric_invchi2_no_trunc_joint
   ```

4. [`scripts/make_paper_outputs.py`](scripts/make_paper_outputs.py) —
   regenerates the figures and LaTeX macros that appear in the paper from a
   completed hierarchical run.

   ```bash
   uv run python scripts/make_paper_outputs.py
   ```

These scripts are GPU-aware (JAX); set `TGRPOP_PLATFORM` and
`TGRPOP_DEVICE_COUNT` to control device selection. The runs reported in the
paper used 4 GPUs.

## Citation

If you use this code (data) in a scientific publication, please cite the paper
and this code (data) release. BibTeX:

```bibtex
@article{Mitman_Isi_Farr_2026,
  author  = {Mitman, Keefe and Isi, Maximiliano and Farr, Will M.},
  title   = {Constraining Gravitational Wave Memory with Hierarchical Inference},
  journal = {arXiv e-prints},
  year    = {2026},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  note    = {LIGO Document LIGO-P2600229},
}

@software{memory_code,
	author       = {Mitman, Keefe and Isi, Maximiliano and Farr, Will M.},
	title        = {farr/Memory: Initial arXiv release},
	month        = may,
	year         = 2026,
	publisher    = {Zenodo},
	version      = {v1.0.0},
	doi          = {10.5281/zenodo.20398639},
	url          = {https://doi.org/10.5281/zenodo.20398639},
}

@dataset{memory_data,
	author       = {Mitman, Keefe and Isi, Maximiliano and Farr, Will M.},
	title        = {Data produced in "Constraining Gravitational Wave Memory with Hierarchical Inference"},
	month        = may,
	year         = 2026,
	publisher    = {Zenodo},
	doi          = {10.5281/zenodo.20347301},
	url          = {https://doi.org/10.5281/zenodo.20347301},
}
```
