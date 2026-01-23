# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project analyzing choice history effects under risk using behavioral data, pupillometry, and fMRI. The project combines data from two datasets (NARPS and Sheng2020) to investigate how previous choices influence current decision-making in risky gambles.

## Repository Structure

- `src/riskhistory/`: Main package containing all analysis code
  - `setup_behavioraldata/`: Python scripts for behavioral data preparation
  - `setup_pupildata/`: R scripts for pupillometry preprocessing
  - `setup_fmridata/`: fMRI data preparation
  - `analysis_acceptglme/`: R-based GLME models for accept/reject choice analysis
  - `analysis_pupillme/`: R-based LME models for pupil analysis
  - `analysis_fmri/`: fMRI analysis (LSA, MVPA, contrast analysis)
  - `config.py` / `config.R`: Configuration files defining data paths
- `scripts/`: Top-level analysis pipeline scripts (currently gitignored)
- `figures/`: Jupyter notebooks for generating manuscript figures (currently gitignored)
- `data/`: Raw and preprocessed data
- `results/`: Analysis outputs (gitignored)
- `statsmodel_specs/`: Statistical model specifications

## Development Setup

This project uses `uv` for Python dependency management:

```bash
# Install dependencies (uses uv.lock)
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate
```

Python version: 3.12+

## Key Commands

### Using Make (Recommended)
```bash
# Show available targets
make help

# Setup behavioral data
make setup_behavioral

# Setup pupil data
make setup_pupildata

# Run accept/reject GLME analysis
make analysis_acceptglme

# Run all analyses
make all_analyses

# Clean results
make clean
```

### Direct Execution
```bash
# Setup behavioral data (Python)
uv run python src/riskhistory/setup_behavioraldata/run_setup_behavioraldata.py

# Run GLME analysis (R) - run from analysis directory
cd src/riskhistory/analysis_acceptglme && Rscript run_analysis_acceptglme.R

# Setup pupil data (R)
cd src/riskhistory/setup_pupildata && Rscript run_setup_pupildata.R
```

## Architecture Details

### Dual-Language Design
The project uses **Python for data processing** and **R for statistical modeling**:
- Python handles data cleaning, preprocessing, and integration across datasets
- R performs generalized/linear mixed effects modeling using lme4/lmerTest
- Both languages share configuration through parallel config files (`config.py` and `config.R`)

### Configuration System
The project uses auto-detected paths with optional environment variable overrides:
- **Python**: `src/riskhistory/config.py` - Auto-detects repo root from file location
- **R**: `src/riskhistory/config.R` - Auto-detects repo root from file location
- **Override**: Set `RISKHISTORY_ROOT` environment variable

Analysis-specific configs extend the base config:
- `analysis_acceptglme/0_config.R` - GLME analysis settings, model lists
- `setup_pupildata/0_config.R` - Pupil preprocessing paths

### Data Flow
1. Raw data (in `data/raw/`) from NARPS and Sheng2020 datasets
2. Python scripts create unified behavioral dataset with choice history variables
3. Preprocessed data saved to `data/preprocessed/`
4. R scripts load preprocessed data and fit hierarchical models
5. Results saved to `results/`

### Choice History Variables
The behavioral data preparation creates lagged choice variables:
- `choice_prev[N]`: Choice made N trials ago (+1 accept, -1 reject)
- `switch`: Whether choice differs from previous trial
- `repeat`: Whether choice matches previous trial
- `switch_button`: Whether button press changed (independent of choice)

These are computed **within runs** to avoid contamination across run boundaries.

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
- Baseline choice model (gain/loss effects)
- Choice history effects (perseveration)
- Lag choice history (choices 2-5 trials back)
- RT history effects
- Value history vs choice history comparisons

Models are numbered sequentially: `1_baselinechoicemodel.R`, `2_choicehistorymodels.R`, etc.

### Pupil Analysis (`analysis_pupillme/`)
Linear mixed-effects models for pupil dilation:
- Response-aligned analysis (`1_pupilmodel_responseAligned.R`)
- Start-aligned analysis (`1_pupilmodel_startAligned.R`)

### fMRI Analysis (`analysis_fmri/`)
- **LSA**: Trial-by-trial beta extraction (`lsa_trial_beta_extraction.py`)
- **MVPA**: Multivariate pattern analysis (`mvpa_classification.py`)
- **Contrast Analysis**: Group-level statistical maps (`analyze_contrasts.py`)
- **VIF Analysis**: Design matrix collinearity checks (`vifs_design.py`)

## Package Dependencies

### Python (see pyproject.toml)
- Data: `pandas`, `numpy`
- Statistics: `statsmodels`, `rpy2` (Python-R interface)
- Visualization: `matplotlib`, `seaborn`
- Notebooks: `jupyter`

### R
- Mixed models: `lme4`, `lmerTest`
- Data manipulation: `dplyr`, `tidyr`
- Model extraction: `broom.mixed`
