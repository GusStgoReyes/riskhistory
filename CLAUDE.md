# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis code for a study of choice history under risk: how the previous accept/reject
choice shapes the current one in a mixed-gamble task, using behavior, pupillometry, and
drift-diffusion modeling. Datasets: NARPS (split into **equalIndifference** and
**equalRange**) and **sheng2020**. `README.md` documents the run order and what every
folder in `src/riskhistory/` does — keep it in sync when adding or renaming steps.

## Repository Structure

```
riskhistory/
├── src/riskhistory/
│   ├── config.py / config.R        # all paths (Python / R)
│   ├── data.py                     # Python loaders for every results file + dataset constants
│   ├── format.py                   # matplotlib defaults (import for side effects)
│   ├── utils.py, utils_pupilviz.py, utils.R
│   ├── setup_behavioraldata/       # Python: trial table + choice history + exclusions
│   ├── setup_pupildata/            # R (eyeris): pupil preprocessing
│   ├── setup_pupilbetas/           # Python: per-trial pupil GLM betas
│   ├── analysis_utils/             # R: shared settings, loaders, regressors, model helpers
│   ├── analysis_acceptglme/        # Python prospect-theory fits + R acceptance GLMEs
│   ├── analysis_rtlme/             # R: RT models
│   ├── analysis_pupiltimeseries/   # R: LME per pupil time bin
│   ├── analysis_pupilbetas/        # R: models of the trial-level pupil betas
│   ├── analysis_pupilbehavior/     # R: pupil x choice-history behavior models
│   ├── analysis_ddm/               # Python, HDDM container only: fit/diagnose/compare/extract
│   ├── analysis_ppcddm/            # Python (pyddm): posterior-predictive simulations
│   ├── analysis_ddmppcbehavior/    # R: models on the simulated behavior
│   └── trash/                      # unused; ignore
├── sbatch/                         # SLURM wrappers (HDDM fits/diagnostics, PPC arrays)
├── scripts/                        # LaTeX table generators
├── figures_official/               # figure notebooks (figureN.ipynb, figureSN.ipynb) -> saved/*.svg
├── data/{raw,preprocessed,derivatives,model_ids}/
└── results/<folder name>/          # analysis outputs (gitignored)
```

## Environments and Commands

- **Python 3.12 via `uv`**: `uv sync` (installs `riskhistory` editable). Run modules with
  `uv run python -m riskhistory.<folder>.<module>`, e.g.
  `uv run python -m riskhistory.setup_behavioraldata.run_setup_behavioraldata`,
  `uv run python -m riskhistory.setup_pupilbetas.run_setup_pupilglm`,
  `uv run python -m riskhistory.analysis_acceptglme.1_fitprospecttheory`.
  Dependency upper bounds in `pyproject.toml` exist for Sherlock's glibc 2.17 — don't raise
  them casually. The system `python3` on this machine is broken; use `.venv/bin/python` or `uv run`.
- **R 4.3+**: every R folder has a `0_config.R`, numbered steps, and a `run_*.R` runner.
  **Run R scripts from their own folder** (`cd src/riskhistory/<folder> && Rscript run_*.R`) —
  they `source()` relative paths.
- **HDDM**: `analysis_ddm` runs only inside the `hcp4715/hddm` Docker image (as
  `hddm_latest.sif` under Singularity on Sherlock), with `PYTHONPATH=<repo>/src`. Its
  `__init__.py` imports `hddm`, so `riskhistory.analysis_ddm` cannot be imported from the uv
  env. Submit with `bash sbatch/{fit_hddm,compute_diagnostics,compare_hddm,extract_posteriors}.bash <dataset>`.
  Models fitted are listed in `data/model_ids/hddm.csv`.
- There is no working Makefile (it is empty).

Pipeline order (details in README): setup_behavioraldata → setup_pupildata → setup_pupilbetas
→ analysis_acceptglme (`1_fitprospecttheory.py` first; the R runner does not call it) →
rtlme / pupiltimeseries / pupilbetas / pupilbehavior → analysis_ddm → analysis_ppcddm →
analysis_ddmppcbehavior.

## Gotchas

- `config.R` **hardcodes** `repo_path` to this machine's clone; `utils.R::get_repo_path()`
  exists but is not used. `config.py` auto-detects the root (`RISKHISTORY_ROOT` overrides).
- DDM results folder: the sbatch jobs and `run_analysis_ppcddm_sampled.py` /
  `subset_posteriors.py` use `results/analysis_ddm_final/`; `data.py` and
  `run_analysis_ppcddm.py` read `results/analysis_ddm/`.
- The sbatch scripts hardcode `REPO_PTH=/scratch/users/gustxsr/riskhistory`.
- Many R results depend on `results/analysis_acceptglme/1_prospecttheory_baseline_betas.csv`
  (subjective values, via `analysis_utils/utils_data.R::load_behavioral_pupil_data`).

## Conventions

- **Choice coding**: `choice_prev[N]` is +1 accept / −1 reject (0 = unavailable), computed
  **within runs** on pre-exclusion data. DDM choice-history parameters (`z_choice_prev1`,
  `v_choice_prev1`) are reported as 2× the coefficient (accept − reject difference).
- **Transitions**: `transition` factor `{accept,reject}_to_{accept,reject}` from
  `analysis_utils/utils_createtransitionregressor.R`; marginaleffects pairwise labels follow
  level order, so normalize direction with `analysis_utils/utils_contrasts.R` or check both
  `"A - B"` / `"B - A"`.
- **Exclusions** (`setup_behavioraldata/run_setup_behavioraldata.py`): trials with RT < 0.2 s
  or > 3.8 s or < 2 previous valid trials; participants with > 30% excluded trials,
  acceptance < 10% or > 90%, or mean RT outside 0.3–3.5 s. Sheng2020 sub 29 is excluded;
  NARPS sub 56's responses are realigned.
- **R statistics**: shared settings in `analysis_utils/0_config.R` (BH correction via
  `p_method`, `GLMER_CONTROL` / `LMER_CONTROL` with bobyqa, seed 42). Each analysis saves
  fixed effects and a convergence report CSV via the shared helpers; scripts define a local
  `save_csv(df, filename)` writing to `analysis_results_path`.
- **Figures**: notebooks in `figures_official/` import `riskhistory.format` and load data only
  through `riskhistory.data`; posteriors are pooled across the 4 chains, summarized as mean
  and 2.5/97.5 percentiles. LaTeX tables are written to `results/*.tex`.
