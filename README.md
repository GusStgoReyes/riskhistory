# Mechanisms of choice history bias in decisions under risk

Analysis code for a study of how the previous choice shapes current decisions in a mixed-gamble accept/reject task, using behavior, pupillometry, and drift-diffusion modeling. Three datasets are analyzed side by side:

- **NARPS**, split into its two conditions, **EqualIndifference** and **EqualRange**
- **Sheng2020**

All code lives in `src/riskhistory/`. Python handles preprocessing, the pupil GLM,
and the DDM; R handles the mixed-effects statistics; the hierarchical DDM (HDDM)
runs inside a container.

To test the pupillometry setup, download the data in [this link](https://drive.google.com/drive/folders/1hbeO82V6PI5qggRbI1eMIt4a5GNxQV7j?usp=sharing). 

---

## How to run

### 1. Environments

There are three, one per kind of step:

| Environment | Used by | Setup |
|---|---|---|
| **Python 3.12** (via [`uv`](https://docs.astral.sh/uv/)) | `setup_behavioraldata`, `setup_pupilbetas`, `analysis_acceptglme/1_fitprospecttheory.py`, `analysis_ppcddm` | `uv sync` from the repo root (installs `riskhistory` in editable mode into `.venv/`) |
| **R 4.3+** | every other `setup_*` / `analysis_*` folder | install the CRAN packages below |
| **HDDM container** | `analysis_ddm` only | see below |

#### R packages

All R packages are on CRAN. Run this once in R:

```r
install.packages(c(
  # mixed models and inference
  "lme4", "lmerTest", "broom.mixed", "marginaleffects", "bayestestR",
  "car", "brglm2",
  # data wrangling
  "dplyr", "tidyr", "tibble", "stringr",
  # pupil preprocessing (setup_pupildata)
  "eyeris",
  # optional: lets run_setup_pupildata.R find its folder when sourced in RStudio
  "rstudioapi"
))
```

The results were produced with R 4.6.1, `lme4` 2.0.6, `marginaleffects` 1.0.0
and `eyeris` 3.3.0. Newer versions should work. If your numbers differ slightly,
check the package versions first.

**Before running any R code**, open `src/riskhistory/config.R` and set `repo_path`
to your clone of this repo. It is hardcoded, and every R analysis takes its paths
from it. (The Python side finds the repo root on its own; set `RISKHISTORY_ROOT`
to override.)

R scripts use relative `source()` calls, so **always run them from their own
folder** (`cd src/riskhistory/<folder> && Rscript <script>.R`).

#### The HDDM container

HDDM needs an old, pinned Python stack, so it runs only inside the
[`hcp4715/hddm`](https://hub.docker.com/r/hcp4715/hddm) Docker image. On a cluster
(e.g. Sherlock), convert it to a Singularity image once:

```bash
singularity build hddm_latest.sif docker://hcp4715/hddm:latest
```

and point `HDDM_SIF` in `sbatch/configs/general.sh` at the `.sif` file. The
container does not have `riskhistory` installed, so every call passes the repo's
`src/` on `PYTHONPATH`:

```bash
# Singularity (what the sbatch scripts run)
singularity exec --env "PYTHONPATH=$PWD/src" hddm_latest.sif \
    python -m riskhistory.analysis_ddm.model_hddm --model_id HDDM_Base ...
```

`riskhistory.analysis_ddm` imports `hddm` on import, so it cannot be imported from
the `uv` environment. The posterior `.npz` files it produces are plain numpy and
can be read anywhere.

### 2. Data

Put the raw data under `data/raw/`:

```
data/raw/narps/behavioral/behav_data.csv
data/raw/narps/pupil/asc/            # EyeLink .asc files
data/raw/sheng2020/behavioral/decision.csv
data/raw/sheng2020/pupil/eye.csv
```

Intermediate files are written to `data/preprocessed/` and `data/derivatives/`;
analysis outputs go to `results/<folder name>/` (both are created as needed).

### 3. Pipeline order

Run the steps in this order. Each step lists what it needs from earlier ones.

```bash
# 1. Behavioral data                                        (needs: raw data)
uv run python -m riskhistory.setup_behavioraldata.run_setup_behavioraldata

# 2. Pupil preprocessing                                    (needs: 1)
cd src/riskhistory/setup_pupildata && Rscript run_setup_pupildata.R
#    optional, after it: Rscript 4_sample_summary_reports.R

# 3. Trial-by-trial pupil GLM betas                         (needs: 1, 2)
uv run python -m riskhistory.setup_pupilbetas.run_setup_pupilglm

# 4. Prospect-theory fits, then acceptance GLMEs            (needs: 1)
uv run python -m riskhistory.analysis_acceptglme.1_fitprospecttheory
cd src/riskhistory/analysis_acceptglme && Rscript run_analysis_acceptglme.R

# 5. RT and pupil mixed models -- any order                 (needs: 1, 4; pupil ones also 2, 3)
cd src/riskhistory/analysis_rtlme            && Rscript run_analysis_rtlme.R
cd src/riskhistory/analysis_pupiltimeseries  && Rscript run_analysispupiltimeseries.R
cd src/riskhistory/analysis_pupilbetas       && Rscript run_analysispupilbetas.R
cd src/riskhistory/analysis_pupilbehavior    && Rscript run_analysis_pupilbehavior.R

# 6. HDDM fits, on the cluster, in the container            (needs: 1, 3)
bash sbatch/fit_hddm.bash <dataset>                  # one array job per model in data/model_ids/hddm.csv
bash sbatch/compute_diagnostics.bash <dataset>       # after the fits; LOO, R-hat, ESS per model
bash sbatch/compare_hddm.bash <dataset> [diag_job_id]  # after diagnostics; waits on the job id if given
bash sbatch/extract_posteriors.bash <dataset>        # after the fits; small .npz posteriors

# 7. Posterior-predictive simulations, on the cluster       (needs: 6)
bash sbatch/run_ppcddm_sampled.sh

# 8. Mixed models on the simulated behavior                 (needs: 7)
cd src/riskhistory/analysis_ddmppcbehavior && Rscript run_analysis_ddmppcbehavior.R
```

`<dataset>` is `equalIndifference`, `equalRange`, or `sheng2020`. Pass `1` as a
second argument to `fit_hddm.bash` for a short test fit.

**Cluster notes.** The sbatch scripts hardcode `REPO_PTH=/scratch/users/gustxsr/riskhistory`
(`run_ppcddm_sampled.sh` also `cd`s there); change it to your own clone. Partition
names, memory, and time limits are set in `sbatch/configs/general.sh` and in each
script header. The HDDM jobs write to `results/analysis_ddm_final/`, which is also
where `run_analysis_ppcddm_sampled.py` reads posteriors from. The local loaders in
`src/riskhistory/data.py` read `results/analysis_ddm/`, so copy or download the
DDM results there to plot them.

---

## What each folder in `src/riskhistory/` does

### Shared modules

| File | Purpose |
|---|---|
| `config.py` / `config.R` | All data and results paths, for Python and R. Every analysis builds its paths from these. |
| `data.py` | Python loaders for every results file the figures use (behavior, GLME fixed effects, pupil contrasts, DDM posteriors/params/diagnostics, PPC simulations), plus dataset constants (`DATASETS`, labels, currencies, runs). |
| `format.py` | Matplotlib defaults (fonts, sizes) applied on import. |
| `utils.py` | Small Python helpers: the gain/loss grid of each dataset, and a bootstrap test for the difference between two Spearman correlations. |
| `utils_pupilviz.py` | Rebuilds pupil time series and epochs from the GLM betas, for plotting. |
| `utils.R` | `get_repo_path()`: finds the repo root from `RISKHISTORY_ROOT` or the sourced file's location. Not currently used by `config.R`. |

### `setup_behavioraldata/` (Python)

Builds `data/preprocessed/behavioral_data.csv`, the single trial-level table that
every later step uses, from the raw NARPS and Sheng2020 files.

- Harmonizes the two datasets, splits NARPS into its two conditions, and fixes
  NARPS subject 56 (responses shifted by one trial).
- Computes the choice-history variables **within each run**, on the full data
  before exclusions, so an excluded trial still provides history for the next
  one (`utils_getchoicehistory.py`). `utils_getchoicehistory.R` holds the
  exponentially decayed choice history that `analysis_acceptglme` sources.
- Trial exclusions: RT < 0.2 s or > 3.8 s, or fewer than 2 previous valid trials.
- Participant exclusions: > 30% trials excluded, acceptance rate < 10% or > 90%,
  mean RT < 0.3 s or > 3.5 s, and Sheng2020 subject 29 (missing trial in run 2).
- Writes sample and exclusion summaries to `results/setup_behavioraldata/`.

Entry point: `run_setup_behavioraldata.py`.

### `setup_pupildata/` (R)

Preprocesses the raw eye-tracking data with `eyeris`.

1. `1_preprocess_narps.R` / `1_preprocess_sheng2020.R` – per-run cleaning and
   filtering; per-subject time series and event timings go to
   `data/derivatives/pupil/{narps,sheng2020}/timeseries/`.
2. `2_combine_datasets.R` – trial epochs aligned to trial start and to the button
   press, merged across datasets into
   `data/preprocessed/pupil_data_aligned{StartTime,ResponseTime}.csv`.
3. `3_exclusions.R` – dataset-specific trial and participant exclusions: adds
   exclusion flags to the aligned epoch files and writes per-participant
   statistics to `results/setup_pupildata/`.
4. `4_sample_summary_reports.R` – optional summary tables; run by hand after the
   main pipeline.

Entry point: `run_setup_pupildata.R` (all steps, or one: `narps`, `sheng2020`,
`combine`, `exclusions`).

### `setup_pupilbetas/` (Python)

Estimates one pupil response amplitude per trial with a GLM: the high-pass-filtered
time series is regressed on per-trial regressors convolved with a pupil response
function (`utils_prf.py`, `utils_glm.py`). Writes
`data/preprocessed/pupilbetas_glm_preprocessed.csv` (the trial-level pupil measure
used by every pupil analysis and the pupil DDMs), plus design VIFs.

Entry point: `run_setup_pupilglm.py`.

### `analysis_utils/` (R)

Code shared by all R analyses: common settings (`0_config.R`: datasets, BH
correction, optimizer settings, seed), data loaders (`utils_data.R`), subjective
values and standardized regressors, the 4-level previous × current choice
`transition` factor, contrast normalization, convergence checks, fixed-effect and
random-effect extraction, and AIC/BIC/Bayes-factor model comparison.

### `analysis_acceptglme/` (Python + R)

Does the previous choice change the probability of accepting a gamble?

1. `1_fitprospecttheory.py` – per-participant prospect-theory fits (loss
   aversion λ, with and without a choice-history term). The baseline betas give
   the subjective values that the later R analyses use. **Run it before the R
   steps**; it is not called by the R runner.
2. `2_choicehistorymodels.R` – acceptance GLMEs: baseline gain/loss, previous
   choice, and the value-history alternatives (previous EV, ΔEV, previous gain,
   previous loss, each with and without previous choice), compared by AIC/BIC/BF.
3. `3_choicehistory_svpartitions.R` – the choice-history effect within
   subjective-value partitions.

Entry point: `run_analysis_acceptglme.R` (steps 2–3).

### `analysis_rtlme/` (R)

Choice history and response times.

1. `1_rtlme_baselinemodel.R` – RT as a function of |subjective value|.
2. `2_rtlme_interactionmodel.R` – adds the previous × current choice `transition`
   factor, with predicted RTs and pairwise contrasts (switch vs. repeat).
3. `3_rtlme_switchpersubject.R` – a per-participant switch effect on RT
   (separate Gamma GLMs, no pooling), used for between-participant correlations.

Entry point: `run_analysis_rtlme.R`.

### `analysis_pupiltimeseries/` (R)

The time course of pupil size by transition: one LME per time bin and dataset,
for both start- and response-aligned epochs, with population-level predictions and
contrasts (`1_timeseries_model.R`, helpers in `utils_timepointmodels.R`).

Entry point: `run_analysispupiltimeseries.R`.

### `analysis_pupilbetas/` (R)

Models of the trial-level pupil betas.

1. `1_regressors_vif.R` – regressor correlations and VIFs.
2. `2_pupilbetas_transition.R` – pupil ~ transition, with accept − reject and
   switch − repeat contrasts.

Entry point: `run_analysispupilbetas.R`.

### `analysis_pupilbehavior/` (R)

Does pupil size modulate the choice-history effect?

1. `1_replicate_acceptmodel.R` / `2_replicate_rtmodel.R` – the main acceptance
   and RT models refit on the pupil-clean subset of trials.
2. `3_pupilacceptmodel.R` – acceptance ~ previous choice × pupil (with and
   without RT), plus the choice-history effect for low vs. high pupil trials.

Entry point: `run_analysis_pupilbehavior.R`.

### `analysis_ddm/` (Python, HDDM container)

Hierarchical drift-diffusion models of where the choice-history effect enters the
decision process: starting point, drift, both, its interaction with gain/loss
valuation, and pupil-modulated versions of each. The models fitted by the sbatch
array are listed in `data/model_ids/hddm.csv` (`model_id,pupil_col`).

| Script | Does |
|---|---|
| `model_hddm.py` | Fits one model on one dataset; saves the full InferenceData (`hddm/`), group and per-participant summaries (`hddm_stats/`, `hddm_params/`). |
| `compute_diagnostics.py` | Per-model LOO, R-hat and ESS, loading one netCDF group at a time to bound memory. |
| `compare_models.py` | Aggregates the diagnostics into LOO model comparisons and convergence tables (`hddm_compare/`). |
| `extract_posteriors.py` | Saves each parameter's posterior samples to a compressed `.npz` (`hddm_posteriors/`) that plain numpy can read. |
| `subset_posteriors.py` | Optionally trims those `.npz` files to a few parameters for download. |
| `utils_*.py` | Data preparation, model definitions, and summary statistics. |

### `analysis_ppcddm/` (Python)

Posterior-predictive checks: simulates choices and RTs for every trial from the
fitted DDMs with `pyddm`, for the baseline and the pupil models.
`run_analysis_ppcddm_sampled.py` draws parameter sets from the posterior (1000 by
default) and averages the simulated per-trial statistics across draws, writing
`results/analysis_ppcddm/{model_id}_simulated_data_sampled.csv`. The older
`run_analysis_ppcddm.py` simulates once from posterior means. Both are SLURM-array
jobs (`--job_index`, `--print_njobs`, `--models`).

### `analysis_ddmppcbehavior/` (R)

Checks whether the simulated behavior reproduces the pupil × choice-history
effect. For each pupil DDM and its no-pupil baseline (starting-point and drift
pairs), it stacks both simulations and fits
`logit(mean_accept) ~ gain + loss + prev_choice × pupil × model` with maximal
random effects, saving fixed effects and a convergence report
(`1_pupilaccept_comparison.R`).

Entry point: `run_analysis_ddmppcbehavior.R`.
