# Choice History Effects Under Risk

A neuroscience research project investigating how previous choices influence current decision-making in risky gambles, combining behavioral data, pupillometry, and fMRI analyses.

## Overview

This project analyzes choice history effects across two datasets:
- **NARPS**: Neuroimaging Analysis Replication and Prediction Study (equalIndifference and equalRange groups)
- **Sheng2020**: Behavioral and pupillometry dataset

The key research question: Do previous accept/reject decisions influence current risky choices, independent of stimulus and value history?

## Installation

### Requirements
- Python 3.12+
- R 4.3+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/poldracklab/riskhistory.git
cd riskhistory

# Install Python dependencies
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate
```

### R Dependencies

Install required R packages:

```r
install.packages(c("lme4", "lmerTest", "dplyr", "tidyr", "broom.mixed"))
```

## Project Structure

```
riskhistory/
├── src/riskhistory/              # Main package
│   ├── setup_behavioraldata/     # Behavioral data preprocessing (Python)
│   ├── setup_pupildata/          # Pupillometry preprocessing (R)
│   ├── setup_pupilbetas/         # Pupil beta extraction via GLM (Python)
│   ├── setup_fmridata/           # fMRI data preparation
│   ├── analysis_acceptglme/      # GLME choice analysis (R)
│   ├── analysis_pupillme/        # Pupil LME analysis per timepoint (R)
│   ├── analysis_pupilbehavior/   # Pupil-behavior relationship analysis (R)
│   ├── analysis_ddm/             # Drift-diffusion modeling (Python/HDDM)
│   ├── config.py                 # Python configuration
│   └── config.R                  # R configuration
├── data/
│   ├── raw/                      # Raw data (not tracked)
│   ├── preprocessed/             # Processed behavioral & pupil data
│   ├── derivatives/              # Derived data products
│   └── model_ids/                # Model identifier mappings
├── results/                      # Analysis outputs (gitignored)
│   ├── analysis_acceptglme/      # Behavioral GLME results
│   ├── analysis_pupillme/        # Pupil LME results
│   ├── analysis_pupilbehavior/   # Pupil-behavior results
│   ├── setup_behavioraldata/     # Behavioral preprocessing logs
│   ├── setup_pupildata/          # Pupil preprocessing logs
│   └── setup_pupilbetas/         # Pupil beta extraction results
├── figures/                      # Figure generation scripts
├── scripts/                      # Top-level pipeline scripts
├── sbatch/                       # SLURM job submission scripts
└── Makefile                      # Build automation
```

## Usage

### Quick Start

```bash
# Show available commands
make help

# Run complete pipeline
make all_analyses
```

### Step-by-Step

1. **Prepare behavioral data**
   ```bash
   make setup_behavioral
   ```

2. **Prepare pupil data**
   ```bash
   make setup_pupildata
   ```

3. **Run statistical analyses**
   ```bash
   make analysis_acceptglme
   ```

### Manual Execution

```bash
# Python scripts
uv run python src/riskhistory/setup_behavioraldata/run_setup_behavioraldata.py
uv run python src/riskhistory/setup_pupilbetas/run_setup_pupilbetas.py

# R scripts (run from analysis directory)
cd src/riskhistory/analysis_acceptglme && Rscript run_analysis_acceptglme.R
cd src/riskhistory/setup_pupildata && Rscript run_setup_pupildata.R
cd src/riskhistory/analysis_pupillme && Rscript run_analysispupillme.R
cd src/riskhistory/analysis_pupilbehavior && Rscript run_analysis_pupilbehavior.R
```

## Analysis Pipeline

### Behavioral Analysis (`analysis_acceptglme/`)

Generalized linear mixed-effects models (GLMEMs) testing:

1. **Baseline Model (M1)**: Gain and loss effects on accept/reject choices
2. **Choice History Model (M2)**: Previous choice predicts current choice
3. **Value History Models (M3-M8)**: Testing alternative explanations (previous subjective value, value change, previous stimuli)
4. **Lag Models (M9-M12)**: Choices 2-5 trials back influence current decisions
5. **Supplementary Models**: Interaction and RT effects

### Pupil Analysis

#### Timepoint Analysis (`analysis_pupillme/`)
Linear mixed-effects models for pupil dilation at each timepoint:
- Response-aligned and stimulus-aligned analyses
- Tests effects of subjective value, entropy, choice, and choice alternation

#### Pupil-Behavior Relationship (`analysis_pupilbehavior/`)
Models examining how trial-level pupil responses relate to choice behavior.

#### Pupil Beta Extraction (`setup_pupilbetas/`)
GLM-based extraction of trial-by-trial pupil response amplitudes using pupil response function (PRF) convolution.

### Drift-Diffusion Modeling (`analysis_ddm/`)

Hierarchical drift-diffusion models (HDDM) testing:
- Whether choice history affects drift rate, boundary separation, or starting point
- Computational mechanisms underlying choice history effects

### fMRI Analysis (`setup_fmridata/`)

Preparation of fMRI event files and data structures for neuroimaging analysis.

## Data

### Raw Data Structure
```
data/raw/
├── narps/
│   ├── behavioral/
│   └── pupil/
└── sheng2020/
    ├── behavioral/
    └── pupil/
```

### Preprocessed Output
- `data/preprocessed/behavioral_data.csv`: Combined behavioral dataset with choice history variables
- `data/preprocessed/pupil_*.csv`: Processed pupillometry data

### Key Variables
| Variable | Description |
|----------|-------------|
| `choice` | Current trial choice (1=accept, 0=reject) |
| `gain_centered` | Centered gain value |
| `loss_centered` | Centered loss value |
| `choice_prev1` | Previous choice (+1=accept, -1=reject) |
| `choice_prev2-5` | Lagged choices 2-5 trials back |
| `switch` | Choice differs from previous trial |
| `rt` | Response time (seconds) |
| `sv` | Subjective value (computed from subject-specific parameters) |
| `entropy` | Decision uncertainty |

## Configuration

The project auto-detects paths from the repository root. Override with environment variables:

```bash
export RISKHISTORY_ROOT=/path/to/riskhistory
```

Each analysis module has its own config file (e.g., `0_config.R`) that extends the base configuration.

## Results

Analysis outputs are saved to `results/<analysis_name>/`:
- Model comparison tables (AIC, BIC, Bayes Factors)
- Fixed effects estimates with confidence intervals
- Random effects summaries
- Convergence diagnostics

## License

[Add license information]

## Citation

[Add citation information]

## Contact

Poldrack Lab, Stanford University
