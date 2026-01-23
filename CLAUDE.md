# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project analyzing choice history effects under risk using behavioral data, pupillometry, and fMRI. The project combines data from two datasets (NARPS and Sheng2020) to investigate how previous choices influence current decision-making in risky gambles.

## Repository Structure

- `src/riskhistory/`: Main package containing all analysis code
  - `behaviorlme/`: R-based linear mixed effects models for behavioral analysis
  - `data/`: Data preparation and preprocessing functions
  - `fmri/`: fMRI analysis scripts (LSA, MVPA, contrast analysis)
  - `pupilprocessing/`: Pupillometry data processing
  - `config.py` / `config.R`: Configuration files defining data paths
- `scripts/`: Top-level analysis pipeline scripts
- `figures/`: Jupyter notebooks for generating manuscript figures
- `data/`: Raw and preprocessed data (gitignored)
- `results/`: Analysis outputs (gitignored)
- `sections/`: Manuscript sections and temporary notebooks

## Development Setup

This project uses `uv` for Python dependency management:

```bash
# Install dependencies (uses uv.lock)
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate
```

Python version: 3.12+

## Data Processing Pipeline

The analysis follows a sequential pipeline:

1. **Data Setup** (`scripts/0-*.py`):
   - `0-setup_behav_data.py`: Prepares behavioral data from both datasets, applies exclusion criteria, creates choice history variables
   - `0-setup_pupil_data.py`: Prepares pupillometry data (currently empty)
   - `0-fmri_event_files.py`: Generates event files for fMRI analysis

2. **Statistical Analysis**:
   - R-based LME models in `src/riskhistory/behaviorlme/run_analysis.R`
   - Models are run sequentially: baseline → choice history → RT history
   - Results saved to `results/` directory

3. **Results Export** (`scripts/1-*.py`):
   - `1-fixedeffects_latextables.py`: Converts fixed effects DataFrames to LaTeX tables

4. **Figure Generation**:
   - Jupyter notebooks in `figures/` directory create manuscript figures
   - Naming convention: `N-description.ipynb` where N indicates figure number

## Key Commands

### Python Data Processing
```bash
# Setup behavioral data (run first)
python scripts/0-setup_behav_data.py

# Generate fMRI event files
python scripts/0-fmri_event_files.py

# Generate LaTeX tables from results
python scripts/1-fixedeffects_latextables.py
```

### R Statistical Analysis
```bash
# Run all behavioral LME analyses
cd src/riskhistory/behaviorlme/
Rscript run_analysis.R
```

## Architecture Details

### Dual-Language Design
The project uses **Python for data processing** and **R for statistical modeling**. This is intentional:
- Python handles data cleaning, preprocessing, and integration across datasets
- R performs linear mixed effects modeling using lme4/lmerTest
- Both languages share configuration through parallel config files (`config.py` and `config.R`)

### Data Flow
1. Raw data (in `data/raw/`) from NARPS and Sheng2020 datasets
2. Python scripts create unified behavioral dataset with choice history variables (prev_choice1-6, switch, repeat)
3. Preprocessed data saved to `data/preprocessed/behavioral_data.csv`
4. R scripts load preprocessed data and fit hierarchical models
5. Results (model comparisons, fixed effects) saved to `results/`
6. Python scripts convert results to publication-ready formats

### Choice History Variables
The behavioral data preparation creates lagged choice variables:
- `choice_prev[N]`: Choice made N trials ago (1 or -1)
- `switch`: Whether choice differs from previous trial
- `repeat`: Whether choice matches previous trial
- `switch_button`: Whether button press changed (independent of choice)

These are computed **within runs** to avoid contamination across run boundaries.

### Subject-Specific Corrections
- **Subject 56 (NARPS)**: Response mapping was inverted; corrected in `data_behavior.py:fix_sub_56()`
- **Subject 29 (Sheng2020)**: Excluded due to missing trial indicator in Run 2 time series

### Exclusion Criteria
Trials are marked for exclusion based on:
- RT < 0.2s or RT > 3.8s
- Fewer than 3 previous valid trials (insufficient choice history)
- Missing gain/loss pairs
- Subject-specific issues (see above)

## Configuration System

The project uses a hierarchical configuration system with auto-detected paths:

### Base Configuration
- **Python**: [config.py](src/riskhistory/config.py) - Auto-detects repo root from file location
- **R**: [config.R](src/riskhistory/config.R) - Auto-detects repo root from file location

Both base configs define standard paths (`raw_data_path`, `preprocessed_data_path`, `results_path`, etc.) and can be overridden via the `RISKHISTORY_ROOT` environment variable.

### Analysis-Specific Configuration
Each analysis module has its own config that sources the base config and adds analysis-specific settings:
- [analysis_acceptglme/0_config.R](src/riskhistory/analysis_acceptglme/0_config.R) - GLME analysis settings, data scaling function, model lists
- [setup_pupildata/config.R](src/riskhistory/setup_pupildata/config.R) - Pupil preprocessing paths, valid subject lists

### Environment Variables
- `RISKHISTORY_ROOT`: Override the auto-detected repository root path
- `RISKHISTORY_PUPIL_DATA`: External pupil data location (for setup_pupildata)
- `RISKHISTORY_PUPIL_OUTPUT_REPO`: Output repository for processed pupil data

## fMRI Analysis Components

- **LSA (Least Squares All)**: Trial-by-trial beta extraction (`lsa_trial_beta_extraction.py`)
- **MVPA**: Multivariate pattern analysis for decoding choice history (`mvpa_classification.py`)
- **Contrast Analysis**: Group-level statistical maps (`analyze_contrasts.py`)
- **VIF Analysis**: Design matrix collinearity checks (`vifs_design.py`)

## Working with Jupyter Notebooks

Figure notebooks often depend on results files being present. Ensure the analysis pipeline has been run before executing figure notebooks:
1. Run data setup scripts
2. Run R analysis (`run_analysis.R`)
3. Execute figure notebooks in order

## Package Dependencies

Key dependencies (see [pyproject.toml](pyproject.toml) for full list):
- Data: `pandas`, `numpy`
- Statistics: `statsmodels`, `rpy2` (Python-R interface)
- Visualization: `matplotlib`, `seaborn`
- Notebooks: `jupyter`