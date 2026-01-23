# Choice History Effects Under Risk

A neuroscience research project investigating how previous choices influence current decision-making in risky gambles, combining behavioral data, pupillometry, and fMRI analyses.

## Overview

This project analyzes choice history effects across two datasets:
- **NARPS**: Neuroimaging Analysis Replication and Prediction Study
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
├── src/riskhistory/           # Main package
│   ├── setup_behavioraldata/  # Behavioral data preprocessing (Python)
│   ├── setup_pupildata/       # Pupillometry preprocessing (R)
│   ├── analysis_acceptglme/   # GLME choice analysis (R)
│   ├── analysis_pupillme/     # Pupil LME analysis (R)
│   ├── analysis_fmri/         # fMRI analysis (Python)
│   ├── config.py              # Python configuration
│   └── config.R               # R configuration
├── data/
│   ├── raw/                   # Raw data (not tracked)
│   └── preprocessed/          # Processed data
├── results/                   # Analysis outputs (not tracked)
├── figures/                   # Figure notebooks (not tracked)
└── Makefile                   # Build automation
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

# R scripts (run from analysis directory)
cd src/riskhistory/analysis_acceptglme
Rscript run_analysis_acceptglme.R
```

## Analysis Pipeline

### Behavioral Analysis

The project uses generalized linear mixed-effects models (GLMEMs) to test:

1. **Baseline Model**: Do gain and loss values predict accept/reject choices?
2. **Choice History Model**: Does the previous choice predict current choice?
3. **Lag Models**: Do choices 2-5 trials back influence current decisions?
4. **Value History Models**: Can value or stimulus history explain apparent choice history effects?

### Pupillometry Analysis

Pupil dilation is analyzed using linear mixed-effects models to examine:
- Choice history effects on pupil responses
- Alignment to stimulus onset vs response

### fMRI Analysis

- **LSA (Least Squares All)**: Trial-by-trial beta extraction
- **MVPA**: Multivariate pattern analysis for decoding choice history
- **Contrast Analysis**: Group-level statistical maps

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
- `data/preprocessed/pupil_data_*.csv`: Processed pupillometry data

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

## Configuration

The project auto-detects paths from the repository root. Override with environment variables:

```bash
export RISKHISTORY_ROOT=/path/to/riskhistory
```

## Results

Analysis outputs are saved to `results/`:
- Model comparison tables (AIC, BIC, Bayes Factors)
- Fixed effects estimates with confidence intervals
- Random effects summaries

## License

[Add license information]

## Citation

[Add citation information]

## Contact

Poldrack Lab, Stanford University
