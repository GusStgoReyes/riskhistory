# Choice History Effects Under Risk

This repository contains the analysis code for a study examining how previous choices influence decision-making under risk, using behavioral data, pupillometry (Sheng2020), and fMRI (NARPS).

## Setup

```bash
uv pip install -e .
source .venv/bin/activate
```

Requires Python 3.12+ and R 4.3+.

---

## Sections

### 1. Behavioral Data Setup (`setup_behavioraldata`)

Loads and preprocesses raw behavioral data from two datasets (NARPS and Sheng2020), computes choice history variables, applies exclusion criteria, and saves a unified behavioral dataset for downstream analyses.

**Inputs** (expected under `data/raw/`):
| File | Description |
|------|-------------|
| `narps/behavioral/behav_data.csv` | NARPS raw behavioral data |
| `sheng2020/behavioral/decision.csv` | Sheng2020 response data |
| `sheng2020/pupil/eye.csv` | Sheng2020 pupil data (used for trial ordering) |

**Outputs**:
| File | Description |
|------|-------------|
| `data/preprocessed/behavioral_data.csv` | Unified behavioral dataset (all subjects, both datasets), including the `response_raw` column (raw response coding, e.g. 1-4 for NARPS, before collapsing to accept/reject) |
| `results/setup_behavioraldata/subject_statistics.csv` | Per-subject statistics (acceptance rate, RT, exclusion flags/reasons per criterion) |
| `results/setup_behavioraldata/trial_exclusion_counts.csv` | Per-subject trial exclusion counts by criterion (pre-participant-exclusion) |
| `results/setup_behavioraldata/trial_exclusion_summary.csv` | Dataset-level trial exclusion summary: mean/sd trials excluded per criterion (RT-low, RT-high, insufficient choice history, missing gain/loss), plus deduplicated total, across subjects (pre-participant-exclusion) |
| `results/setup_behavioraldata/participant_exclusion_summary.csv` | Dataset-level participant exclusion summary: count of participants excluded per criterion, plus deduplicated total |
| `results/setup_behavioraldata/final_sample_summary.csv` | Dataset-level sample size and behavior statistics after participant exclusion: subjects retained, valid trials (mean/min/max), acceptance rate (mean/sd), RT (mean/sd) |

**Exclusion criteria** (configurable at the top of `run_setup_behavioraldata.py`):

*Trial-level:*
- RT < 0.2 s or RT > 3.8 s
- Fewer than 2 valid preceding trials (choice history unavailable)
- Missing gain/loss pair

*Participant-level:*
- > 30% of trials excluded
- Acceptance rate < 10% or > 90%
- Mean RT < 0.3 s or > 3.5 s
- Manual exclusion: Sheng2020 subject 29 (missing trial indicator in Run 2)

Trial-level exclusion criteria are evaluated independently (not cumulatively) for the per-criterion counts in `trial_exclusion_counts.csv`/`trial_exclusion_summary.csv`, so a trial (or participant, for `participant_exclusion_summary.csv`) failing multiple criteria is counted once under each criterion but only once in the deduplicated total.

**How to run:**

```bash
uv run python src/riskhistory/setup_behavioraldata/run_setup_behavioraldata.py
```

---

### 2. Pupil Data Setup (`setup_pupildata`)

Preprocesses raw pupil data for both datasets, extracts trial-aligned epochs, applies data quality exclusions, and saves a unified pupil dataset for downstream analyses. Requires the behavioral data from step 1.

**Dependencies:** `eyeris`, `dplyr`, `stringr`, `tidyr` R packages.

**Inputs** (expected under `data/raw/`):
| File | Description |
|------|-------------|
| `narps/pupil/asc/*.asc` | NARPS raw eye-tracking files (one per subject-run) |
| `sheng2020/pupil/eye.csv` | Sheng2020 pupil time series |
| `data/preprocessed/behavioral_data.csv` | Output of step 1 (used for behavioral exclusion flags) |

**Processing steps:**
1. **NARPS**: Load `.asc` files → deblink → detransient → interpolate → low-pass filter → detrend → z-score → bin at 8 Hz → extract start- and response-aligned epochs
2. **Sheng2020**: Load CSV → fill missing samples → convert to eyeris format → detransient (no deblink) → same pipeline as NARPS
3. **Combine**: Merge NARPS and Sheng2020 epoch data; assign NARPS condition labels (odd sub = equalIndifference, even sub = equalRange)
4. **Exclusions**: Apply trial- and subject-level data quality thresholds (per dataset); merge with behavioral exclusions from step 1
5. **Summary statistics**: Compute per-subject behavioral statistics for pupil-included participants

**Key parameters** (in `0_config.R`):
| Parameter | Value | Description |
|-----------|-------|-------------|
| `OUTPUT_SAMPLING_RATE` | 8 Hz | Final epoch sampling rate |
| `START_EPOCH_LIMITS` | [0, 4] s | Stimulus-onset aligned window |
| `RESPONSE_EPOCH_LIMITS` | [-2, 2] s | Response-aligned window |
| `BASELINE_PERIOD` | [-0.5, 0] s | Baseline subtraction window |
| `MAX_NA_PERCENTAGE` | 90% | Skip subject-run if >90% missing |
| `MAX_EXCLUDED_TRIALS_PCT` | 30% | Exclude subject if >30% of trials excluded |

**Outputs**:
| File | Description |
|------|-------------|
| `data/preprocessed/pupil_data_alignedStartTime.csv` | Stimulus-onset aligned epochs with exclusion flags |
| `data/preprocessed/pupil_data_alignedResponseTime.csv` | Response-aligned epochs with exclusion flags |
| `results/setup_pupildata/pupil_data_statistics.csv` | Per-trial data quality statistics |
| `results/setup_pupildata/pupil_data_statistics_with_exclusions.csv` | Trial statistics with exclusion flags |
| `results/setup_pupildata/pupil_subject_exclusions.csv` | Subject-level exclusion summary |
| `results/setup_pupildata/summary_exclusion.txt` | Human-readable exclusion report |
| `results/setup_pupildata/4_summary_statistics.csv` | Dataset-level behavioral summary for included participants |
| `data/derivatives/pupil/{narps,sheng2020}/timeseries/` | Per-subject filtered time series (25 Hz) |

**How to run:**

```bash
# Run full pipeline (from repo root)
Rscript src/riskhistory/setup_pupildata/run_setup_pupildata.R

# Run individual steps
Rscript src/riskhistory/setup_pupildata/run_setup_pupildata.R narps        # NARPS preprocessing only
Rscript src/riskhistory/setup_pupildata/run_setup_pupildata.R sheng2020    # Sheng2020 preprocessing only
Rscript src/riskhistory/setup_pupildata/run_setup_pupildata.R combine      # Combine datasets only
Rscript src/riskhistory/setup_pupildata/run_setup_pupildata.R exclusions   # Apply exclusions only
Rscript src/riskhistory/setup_pupildata/run_setup_pupildata.R summary      # Summary statistics only
```

> **Note:** Steps must be run in order (narps/sheng2020 → combine → exclusions → summary). Individual steps can be rerun after fixing intermediate files without rerunning the full pipeline. The runner can be called from any working directory.

---

### 3. Pupil Beta Extraction (`setup_pupilbetas`)

Fits a trial-wise LS-A GLM to each subject's preprocessed pupil time series to extract per-trial amplitude betas. A separate script extracts the pre-stimulus baseline. Requires steps 1 and 2.

**Inputs:**
| File | Description |
|------|-------------|
| `data/preprocessed/behavioral_data.csv` | Output of step 1 |
| `results/setup_pupildata/pupil_data_statistics_with_exclusions.csv` | Output of step 2 |
| `data/derivatives/pupil/{narps,sheng2020}/timeseries/` | Filtered time series from step 2 |

**Key parameters** (in `run_setup_pupilglm.py` / `run_setup_pupilbaseline.py`):
| Parameter | Value | Description |
|-----------|-------|-------------|
| `HIGHPASS_CUTOFF` | 0.05 Hz | High-pass filter applied before GLM |
| `LOWPASS_CUTOFF` | 0.05 Hz | Low-pass filter for baseline extraction |
| `PRF_WIDTH` / `PRF_TIME_TO_PEAK` | 10.1 / 930 ms | Pupil response function (Hoeks & Levelt) |
| `alphas` | `logspace(-5, -1, 25)` | Ridge regression regularisation grid |

**Outputs:**
| File | Description |
|------|-------------|
| `data/preprocessed/pupilbetas_glm_preprocessed.csv` | Per-trial GLM beta (within-run z-scored), ridge alpha, exclusion flag |
| `data/preprocessed/pupilbaseline_preprocessed.csv` | Per-trial pre-stimulus baseline (within-run z-scored), exclusion flag |
| `results/setup_pupilbetas/pupilbetas_glm_results.csv` | Full GLM output (all regressors, all runs) |
| `results/setup_pupilbetas/design_vif_per_run.csv` | VIF for decision regressors per subject-run |
| `results/setup_pupilbetas/design_vif_summary.csv` | Mean VIF by dataset |

**How to run:**

```bash
# GLM betas (LS-A per trial)
uv run python src/riskhistory/setup_pupilbetas/run_setup_pupilglm.py

# Pre-stimulus baseline
uv run python src/riskhistory/setup_pupilbetas/run_setup_pupilbaseline.py
```

---

### 4. Behavioral GLME Analysis (`analysis_acceptglme`)

Fits a series of generalized linear mixed-effects models (binomial) to test whether and how previous choices, previous values, and previous stimuli influence current accept/reject decisions. Requires step 1.

**Inputs:**
| File | Description |
|------|-------------|
| `data/preprocessed/behavioral_data.csv` | Output of step 1 |

**Outputs:**
| File | Description |
|------|-------------|
| `results/analysis_acceptglme/1_baselinebehavior_betas.csv` | Per-subject gain/loss/intercept betas from M1 |
| `results/analysis_acceptglme/2_behavior_model_comparison_{1,2,3}.csv` | AIC/BIC/BF comparisons for M1–M8 |
| `results/analysis_acceptglme/2_behavior_fixed_effects.csv` | Fixed effects for M1–M8 (BH-corrected) |
| `results/analysis_acceptglme/2_choicehistory_betas.csv` | Per-subject choice history random slopes from M2 |
| `results/analysis_acceptglme/3_supplementary_model_comparison.csv` | AIC/BIC/BF for supplementary interaction model |
| `results/analysis_acceptglme/3_supplementary_fixed_effects.csv` | Fixed effects for supplementary model |
| `results/analysis_acceptglme/4_model_convergence_report.csv` | Convergence and singularity flags for all models |

**How to run:**

```bash
# Full analysis (M1–M8 + supplementary + decay-rate profile)
cd src/riskhistory/analysis_acceptglme && Rscript run_analysis_acceptglme.R
```

> **Note:** Must be run from its directory due to relative `source()` calls. The decay-rate profile sweep (`3b_choicehistory_profile.R`) runs automatically as part of the pipeline and does not need to be invoked separately.

---

### 5. Pupil Timeseries Analysis (`analysis_pupiltimeseries`)

Fits a linear mixed-effects model per timepoint across the trial epoch (start- and response-aligned) to test whether pupil dilation tracks value, entropy, RT, and the choice-history `transition` factor (current x previous choice). Estimated marginal means and pairwise contrasts for `transition` are computed per timepoint via `marginaleffects` (population-level, re.form = NA). Requires steps 1–2.

Trial-wise pupil GLM betas (not per-timepoint) are handled separately in `analysis_pupilbetas`.

**Inputs:**
| File | Description |
|------|-------------|
| `data/preprocessed/behavioral_data.csv` | Output of step 1 |
| `data/preprocessed/pupil_data_alignedStartTime.csv` | Output of step 2 |
| `data/preprocessed/pupil_data_alignedResponseTime.csv` | Output of step 2 |
| `results/analysis_acceptglme/1_baselinebehavior_betas.csv` | Output of step 4 (for SV computation) |

**Outputs:**
| File | Description |
|------|-------------|
| `results/analysis_pupiltimeseries/1_timeseries_start_fixed_effects.csv` | Per-timepoint fixed effects, start-aligned |
| `results/analysis_pupiltimeseries/1_timeseries_start_emmeans.csv` | Per-timepoint transition emmeans, start-aligned |
| `results/analysis_pupiltimeseries/1_timeseries_start_contrasts.csv` | Per-timepoint transition contrasts, start-aligned |
| `results/analysis_pupiltimeseries/1_timeseries_response_fixed_effects.csv` | Per-timepoint fixed effects, response-aligned |
| `results/analysis_pupiltimeseries/1_timeseries_response_emmeans.csv` | Per-timepoint transition emmeans, response-aligned |
| `results/analysis_pupiltimeseries/1_timeseries_response_contrasts.csv` | Per-timepoint transition contrasts, response-aligned |

**How to run:**

```bash
cd src/riskhistory/analysis_pupiltimeseries && Rscript run_analysispupiltimeseries.R
```

---

### 6. Drift-Diffusion Modeling (`analysis_ddm`)

Fits hierarchical DDMs (HDDM) to test whether choice history influences the drift rate or starting point of the accumulation process. Requires step 1 (and step 3 for pupil-linked models).

**Available models** (see `utils_model.py` for full specifications):

| Model ID | Boundary | History effect |
|----------|----------|----------------|
| `HDDM_Collapse` | Collapsing (angle) | None (baseline) |
| `HDDM_Collapse_Lag1-z0` | Collapsing | Starting point (`z`) |
| `HDDM_Collapse_Lag1-v0` | Collapsing | Drift rate (`v`) |
| `HDDM_Collapse_Lag1-z0_v0` | Collapsing | Both `z` and `v` |
| `HDDM_Base` | Fixed | None (baseline) |
| `HDDM_Base_Lag1-stp` | Fixed | Starting point (`z`) |
| `HDDM_Base_Lag1-drift` | Fixed | Drift rate (`v`) |
| `HDDM_Base_Lag1-stim` | Fixed | Stimulus × history interaction on `v` |
| `HDDM_Base_Lag1-full` | Fixed | Both `z` and `v` with interaction |

**Outputs** (per model-dataset pair):
| File | Description |
|------|-------------|
| `results/analysis_ddm/hddm/{model_id}_{dataset}.nc` | Full posterior (InferenceData / NetCDF) |
| `results/analysis_ddm/hddm_stats/{model_id}_{dataset}_model_stats.csv` | Group-level parameters + DIC/AIC/BIC |
| `results/analysis_ddm/hddm_params/{model_id}_{dataset}_subj_params.csv` | Per-subject posterior means and SDs |
| `results/analysis_ddm/hddm_compare/{dataset}_model_comparison_waic.csv` | WAIC model comparison table |
| `results/analysis_ddm/hddm_compare/{dataset}_{rhat,ess}.csv` | Convergence diagnostics |
| `results/analysis_ddm/hddm_posteriors/{model_id}_{dataset}_posterior.npz` | Raw posterior samples per parameter (numpy-only, no hddm/arviz needed to read) |

**How to run:**

```bash
# Fit a single model
uv run python -m riskhistory.analysis_ddm.model_hddm \
    --model_id HDDM_Collapse \
    --dataset_name equalIndifference \
    --data_pth data/preprocessed/behavioral_data.csv \
    --pupil_pth data/preprocessed/pupilbetas_glm_preprocessed.csv \
    --results_pth results/analysis_ddm

# Quick test run (4 subjects, 10 iterations)
uv run python -m riskhistory.analysis_ddm.model_hddm --test \
    --model_id HDDM_Collapse \
    --dataset_name equalIndifference \
    --data_pth data/preprocessed/behavioral_data.csv \
    --pupil_pth data/preprocessed/pupilbetas_glm_preprocessed.csv \
    --results_pth results/analysis_ddm

# Compute per-model diagnostics (run once per fitted model)
uv run python -m riskhistory.analysis_ddm.compute_diagnostics \
    --model_id HDDM_Collapse \
    --dataset_name equalIndifference \
    --results_pth results/analysis_ddm

# Compare models (run after compute_diagnostics for all models)
uv run python -m riskhistory.analysis_ddm.compare_models \
    --models HDDM_Collapse HDDM_Collapse_Lag1-z0 HDDM_Collapse_Lag1-v0 HDDM_Collapse_Lag1-z0_v0 \
    --dataset_name equalIndifference \
    --results_pth results/analysis_ddm

# Extract posterior parameter samples into a small, numpy-only .npz for local plotting
uv run python -m riskhistory.analysis_ddm.extract_posteriors \
    --model_id HDDM_Collapse \
    --dataset_name equalIndifference \
    --results_pth results/analysis_ddm
```

> **Note:** Each model fit is computationally intensive (default: 10,000 samples × 4 chains). Models are intended to be run as separate SLURM jobs — see `sbatch/` for job specifications.

> **Note:** `extract_posteriors.py` opens only the `posterior` netCDF4 group of the `.nc` file, never loading the much larger `log_likelihood`/`observed_data` groups, so it's cheap enough to run on a login node without SLURM. An array-job wrapper is also available at `sbatch/extract_posteriors.bash` for running it across all models at once.

---

### 7. Posterior Predictive Checks — DDM (`analysis_ppcddm`)

Simulates trial-level choice and RT distributions from the five fixed-boundary HDDM models using each subject's posterior mean parameters. pyddm is used for the forward simulation. Requires the HDDM outputs from step 6.

**Models simulated:**

| Label | HDDM model ID | History mechanism |
|-------|---------------|-------------------|
| Baseline | `HDDM_Base` | None |
| Starting Point Bias | `HDDM_Base_Lag1-stp` | Starting point (`z`) |
| Drift Bias | `HDDM_Base_Lag1-drift` | Drift rate (`v`) |
| Stimulus Bias | `HDDM_Base_Lag1-stim` | Stimulus × history on `v` |
| Full | `HDDM_Base_Lag1-full` | Both `z` and `v` with interaction |

**Inputs** (per model-dataset pair):

| File | Description |
|------|-------------|
| `data/preprocessed/behavioral_data.csv` | Output of step 1 (trial conditions) |
| `results/analysis_ddm/hddm_params/{model_id}_{dataset}_subj_params.csv` | Per-subject posterior means from step 6 |
| `results/analysis_ddm/hddm_stats/{model_id}_{dataset}_model_stats.csv` | Group-level posterior means from step 6 |

**Outputs** (one CSV per model, appended across subjects):

| File | Description |
|------|-------------|
| `results/analysis_ppcddm/{model_id}_simulated_data.csv` | Simulated mean accept rate, mean RT, and repeat probability per trial condition and subject |

**How to run:**

```bash
# Print total number of jobs (for SLURM array sizing)
uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm --print_njobs

# Run a single subject (SLURM array job)
uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm --job_index $SLURM_ARRAY_TASK_ID

# Run all subjects sequentially (local, slow)
uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm
```

> **Note:** Each job simulates all five models for one (dataset, subject) pair. Concurrent writes across array jobs are safe — output CSVs are protected by a file lock. Run after all HDDM fits (step 6) are complete.

---
