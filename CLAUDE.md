# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project analyzing choice history effects under risk using behavioral data, pupillometry, and fMRI. The project combines data from two datasets (NARPS and Sheng2020) to investigate how previous choices influence current decision-making in risky gambles.

## Repository Structure

```
riskhistory/
├── src/riskhistory/              # Main package
│   ├── setup_behavioraldata/     # Behavioral data preprocessing (Python)
│   ├── setup_pupildata/          # Pupillometry preprocessing (R)
│   ├── setup_pupilbetas/         # Pupil beta extraction via GLM (Python)
│   ├── setup_fmridata/           # fMRI data preparation
│   ├── analysis_acceptglme/      # GLME choice analysis (R)
│   ├── analysis_pupillme/        # Pupil LME per timepoint (R)
│   ├── analysis_pupilbehavior/   # Pupil-behavior relationship (R)
│   ├── analysis_ddm/             # Drift-diffusion modeling (Python)
│   ├── config.py                 # Python configuration
│   └── config.R                  # R configuration
├── data/
│   ├── raw/                      # Raw data (not tracked)
│   ├── preprocessed/             # Processed data
│   ├── derivatives/              # Derived data products
│   └── model_ids/                # Model identifier mappings
├── results/                      # Analysis outputs (gitignored)
├── figures/                      # Figure generation scripts
├── scripts/                      # Top-level pipeline scripts
├── sbatch/                       # SLURM job model specifications
```

## Development Setup

This project uses `uv` for Python dependency management:

```bash
# Install dependencies
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate
```

Python version: 3.12+
R version: 4.3+

## Key Commands

### Using Make (Recommended)
```bash
make help                    # Show available targets
make setup_behavioral        # Setup behavioral data (Python)
make setup_pupildata         # Setup pupil data (R)
make analysis_acceptglme     # Run accept/reject GLME analysis (R)
make all_analyses            # Run all analyses
make clean                   # Remove generated results
```

### Direct Execution
```bash
# Python scripts
uv run python src/riskhistory/setup_behavioraldata/run_setup_behavioraldata.py
uv run python src/riskhistory/setup_pupilbetas/run_setup_pupilbetas.py

# R scripts (run from their directories)
cd src/riskhistory/analysis_acceptglme && Rscript run_analysis_acceptglme.R
cd src/riskhistory/setup_pupildata && Rscript run_setup_pupildata.R
cd src/riskhistory/analysis_pupillme && Rscript run_analysispupillme.R
cd src/riskhistory/analysis_pupilbehavior && Rscript run_analysis_pupilbehavior.R
```

## Architecture Details

### Dual-Language Design
- **Python**: Data preprocessing, GLM-based beta extraction, DDM modeling
- **R**: Statistical modeling using lme4/lmerTest for mixed-effects models
- Both languages share configuration through parallel config files (`config.py` and `config.R`)

### Configuration System
Auto-detected paths with optional environment variable overrides:
- **Python**: `src/riskhistory/config.py`
- **R**: `src/riskhistory/config.R`
- **Override**: Set `RISKHISTORY_ROOT` environment variable

Analysis-specific configs extend the base config:
- `analysis_acceptglme/0_config.R`
- `analysis_pupillme/0_config.R`
- `analysis_pupilbehavior/0_config.R`
- `setup_pupildata/0_config.R`

### Data Flow
1. Raw data (in `data/raw/`) from NARPS and Sheng2020 datasets
2. Python scripts create unified behavioral dataset with choice history variables
3. Preprocessed data saved to `data/preprocessed/`
4. R scripts load preprocessed data and fit hierarchical models
5. Results saved to `results/<analysis_name>/`

### Choice History Variables
Computed **within runs** to avoid contamination across run boundaries:
- `choice_prev[N]`: Choice made N trials ago (+1 accept, -1 reject)
- `switch`: Whether choice differs from previous trial
- `repeat`: Whether choice matches previous trial
- `switch_button`: Whether button press changed (independent of choice)

### Subject-Specific Corrections
- **Subject 56 (NARPS)**: Response mapping was inverted; corrected in preprocessing
- **Subject 29 (Sheng2020)**: Excluded due to missing trial indicator

### Exclusion Criteria
Trials are excluded based on:
- RT < 0.2s or RT > 3.8s
- Fewer than 3 previous valid trials (insufficient choice history)
- Missing gain/loss pairs

## Analysis Components

### Behavioral GLME Analysis (`analysis_acceptglme/`)
Generalized linear mixed-effects models testing:
- `1_baselinechoicemodel.R`: Baseline gain/loss effects (M1)
- `2_choicehistorymodels.R`: Choice history and value history models (M2-M8)
- `3_lagchoicehistorymodels.R`: Lagged choices 2-5 trials back (M9-M12)
- `4_rthistorymodels.R`: Response time history effects
- `5_supplementarymodels.R`: Interaction and additional models

Key utility files:
- `utils_modelcomparison.R`: AIC/BIC/BF model comparison
- `utils_savefixedeffects.R`: Extract and save fixed effects
- `utils_addsubjectivevalues.R`: Compute trial-by-trial subjective values

### Pupil Timepoint Analysis (`analysis_pupillme/`)
Linear mixed-effects models per timepoint:
- `2_pupilmodel_startAligned.R`: Stimulus-onset aligned analysis
- `2_pupilmodel_responseAligned.R`: Response-aligned analysis
- `utils_timepointmodels.R`: Functions for fitting models across timepoints

Model formula:
```r
pupil_size ~ sv + entropy + logRT + choice + switch + (1 + switch + sv || sub)
```

### Pupil-Behavior Analysis (`analysis_pupilbehavior/`)
Models examining pupil-behavior relationships:
- `1_pupilchoicemodel.R`: How pupil dilation relates to choice
- `utils_pupilpertrial.R`: Per-trial pupil metrics extraction

### Pupil Beta Extraction (`setup_pupilbetas/`)
GLM-based trial-by-trial amplitude extraction:
- `run_setup_pupilbetas.py`: Main runner
- `utils_glm.py`: GLM fitting with PRF convolution
- `utils_prf.py`: Pupil response function definitions
- `utils_io.py`: Data I/O utilities

### Drift-Diffusion Modeling (`analysis_ddm/`)
Hierarchical DDM using HDDM:
- `model_hddm.py`: Model definitions
- `utils_data.py`: Data preparation for HDDM
- `utils_model.py`: Model fitting utilities
- `utils_stats.py`: Statistical analysis of DDM parameters

### Data Setup

#### Behavioral Data (`setup_behavioraldata/`)
- `run_setup_behavioraldata.py`: Main preprocessing script
- `utils_behavioraldata.py`: Data loading and transformation functions

#### Pupil Data (`setup_pupildata/`)
- `1_preprocess_narps.R`: NARPS pupil preprocessing
- `1_preprocess_sheng2020.R`: Sheng2020 pupil preprocessing
- `2_combine_datasets.R`: Merge datasets
- `3_exclusions.R`: Apply exclusion criteria
- `utils_eyeris.R`: eyeris package utilities

## Package Dependencies

### Python (pyproject.toml)
- Data: `pandas`, `numpy`
- Statistics: `statsmodels`, `rpy2`
- Visualization: `matplotlib`, `seaborn`
- Notebooks: `jupyter`

### R
- Mixed models: `lme4`, `lmerTest`
- Data manipulation: `dplyr`, `tidyr`
- Model extraction: `broom.mixed`

## Testing and Validation

Models are validated through:
- Convergence checks (`utils_checkmodelconverge.R`)
- VIF analysis for multicollinearity
- Random effects structure comparison
- Cross-dataset replication (equalIndifference, equalRange, Sheng2020)

## Notes

- The Makefile references `setup_pupildata_v2` which should be `setup_pupildata`
- R scripts must be run from their respective directories due to relative path usage
- Results directories are created automatically by analysis scripts
