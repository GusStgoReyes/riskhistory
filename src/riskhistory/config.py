# =============================================================================
# Choice History Under Risk: Config File for Risk History Analyses in Python
# =============================================================================
import os
from pathlib import Path

# Auto-detect repo root: config.py is at src/riskhistory/config.py
# so repo root is 2 levels up. Can override with RISKHISTORY_ROOT env var.
_default_repo_pth = Path(__file__).resolve().parent.parent.parent
repo_pth = Path(os.environ.get("RISKHISTORY_ROOT", _default_repo_pth))

# =============================================================================
# Base directory paths
# =============================================================================
raw_data_pth = repo_pth / "data" / "raw"
preprocessed_data_pth = repo_pth / "data" / "preprocessed"
derivatives_pth = repo_pth / "data" / "derivatives"
results_pth = repo_pth / "results"

# =============================================================================
# Raw data paths
# =============================================================================
narps_behavioral_data_pth = raw_data_pth / "narps" / "behavioral" / "behav_data.csv"
sheng2020_behavioral_data_pth = raw_data_pth / "sheng2020" / "behavioral" / "decision.csv"
sheng2020_pupil_data_pth = raw_data_pth / "sheng2020" / "pupil" / "eye.csv"

# =============================================================================
# Preprocessed/output paths
# =============================================================================
behavioral_data_pth = preprocessed_data_pth / "behavioral_data.csv"
subject_statistics_pth = results_pth / "setup_behavioraldata" / "subject_statistics.csv"
behavioral_summary_pth = results_pth / "setup_behavioraldata" / "behavioral_summary.txt"