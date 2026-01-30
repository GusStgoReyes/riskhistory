# =============================================================================
# Pupil Beta Extraction: I/O Utilities
# =============================================================================
# Functions for loading timeseries and timing data.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set

from riskhistory.config import derivatives_pth, results_pth


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_CONFIG = {
    "sheng2020": {
        "n_runs": 2,
        "trials_per_run": 100,
        "expected_sampling_rate": 125,  # Hz
        "hardcoded_invalid_subjects": [29],
    },
    "equalIndifference": {
        "n_runs": 4,
        "trials_per_run": 64,
        "expected_sampling_rate": 500,  # Hz (could also be 1000)
        "hardcoded_invalid_subjects": [],
    },
    "equalRange": {
        "n_runs": 4,
        "trials_per_run": 64,
        "expected_sampling_rate": 500,  # Hz (could also be 1000)
        "hardcoded_invalid_subjects": [],
    }
}


def get_analysis_dataset(source_dataset: str, sub_id: int) -> str:
    """
    Get the analysis dataset name for a subject.

    For narps: odd subjects -> equalIndifference, even -> equalRange
    For sheng2020: always sheng2020

    Parameters
    ----------
    source_dataset : str
        Source dataset ("sheng2020" or "narps")
    sub_id : int
        Subject ID

    Returns
    -------
    str
        Analysis dataset name
    """
    if source_dataset == "sheng2020":
        return "sheng2020"
    elif source_dataset == "narps":
        return "equalIndifference" if sub_id % 2 == 1 else "equalRange"
    else:
        raise ValueError(f"Unknown dataset: {source_dataset}")


def get_timeseries_dir(dataset: str) -> Path:
    """Get the timeseries directory for a dataset."""
    if dataset in ["equalIndifference", "equalRange"]:
        dataset = "narps"
    return derivatives_pth / "pupil" / dataset / "timeseries"


def get_timings_path(dataset: str) -> Path:
    """Get the path to the start timings file for a dataset."""
    if dataset in ["equalIndifference", "equalRange"]:
        dataset = "narps"
    return get_timeseries_dir(dataset) / f"{dataset}_start_timings.csv"


def get_exclusions_path() -> Path:
    """Get path to subject exclusions file."""
    return results_pth / "setup_pupildata" / "pupil_subject_exclusions.csv"


# =============================================================================
# Exclusions Loading
# =============================================================================
def load_subject_exclusions() -> pd.DataFrame:
    """
    Load subject exclusions from the pupil preprocessing results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [sub, dataset, exclude_subject_startAligned,
        exclude_subject_responseAligned]

    Raises
    ------
    FileNotFoundError
        If the exclusions file does not exist
    """
    exclusions_path = get_exclusions_path()

    if not exclusions_path.exists():
        raise FileNotFoundError(
            f"Exclusions file not found: {exclusions_path}. "
            f"Run the pupil preprocessing pipeline first."
        )

    df = pd.read_csv(exclusions_path)
    return df[['sub', 'dataset', 'exclude_subject_startAligned',
               'exclude_subject_responseAligned']]


def get_valid_subjects(source_dataset: str) -> Set[int]:
    """
    Get set of subjects to include for a dataset.

    A subject is included if:
    - They are not excluded in either startAligned or responseAligned

    Parameters
    ----------
    source_dataset : str
        Source dataset ("sheng2020" or "equalIndifference" or "equalRange")

    Returns
    -------
    Set[int]
        Set of subject IDs to exclude
    """
    included = []
    # Load exclusions file
    exclusions_df = load_subject_exclusions()
    # Filter by dataset
    mask = exclusions_df['dataset'] == source_dataset

    dataset_exclusions = exclusions_df[mask]

    # Include if both columns are False 
    for _, row in dataset_exclusions.iterrows():
        if not row['exclude_subject_startAligned'] and not row['exclude_subject_responseAligned']:
            included.append(int(row['sub']))

    return included


# =============================================================================
# Outlier Removal
# =============================================================================
def remove_outliers(
    values: np.ndarray,
    z_threshold: float = 3.0
) -> np.ndarray:
    """
    Remove outliers by setting values beyond z_threshold standard deviations to NaN.

    Parameters
    ----------
    values : np.ndarray
        Input values
    z_threshold : float
        Z-score threshold for outlier detection (default: 3.0)

    Returns
    -------
    np.ndarray
        Values with outliers set to NaN
    """
    values = values.astype(float).copy()

    # Compute mean and std ignoring existing NaNs
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)

    # Avoid division by zero
    if std_val == 0 or np.isnan(std_val):
        return values

    # Compute z-scores
    z_scores = np.abs((values - mean_val) / std_val)

    # Set outliers to NaN
    values[z_scores > z_threshold] = np.nan

    return values

# =============================================================================
# Timeseries Loading
# =============================================================================
def load_timeseries(
    sub_id: int,
    run_id: int,
    dataset: str,
    remove_outliers_z: float = 0
) -> pd.DataFrame:
    """
    Load pupil timeseries for a subject-run with outlier removal.

    Parameters
    ----------
    sub_id : int
        Subject ID
    run_id : int
        Run ID
    dataset : str
        Dataset name ("sheng2020" or "narps")
    remove_outliers_z : float
        Z-score threshold for outlier removal. Values beyond this threshold
        are set to NaN. Default is 3.0. Set to 0 or None to disable.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [time_secs, pupil_filtered, eye_x, eye_y]
        Outliers in pupil_filtered, eye_x, and eye_y are set to NaN.

    Raises
    ------
    FileNotFoundError
        If the timeseries file does not exist
    """
    timeseries_dir = get_timeseries_dir(dataset)
    filename = f"sub-{sub_id:03d}_run-{run_id}_timeseries.csv"
    filepath = timeseries_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Timeseries file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Apply outlier removal if threshold is specified
    if remove_outliers_z and remove_outliers_z > 0:
        for col in ['pupil_filtered', 'eye_x', 'eye_y']:
            if col in df.columns:
                df[col] = remove_outliers(df[col].values, z_threshold=remove_outliers_z)

    return df


def estimate_sampling_rate(timeseries: pd.DataFrame) -> float:
    """
    Estimate sampling rate from timeseries data.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Timeseries with time_secs column

    Returns
    -------
    float
        Estimated sampling rate in Hz
    """
    time_diff = np.diff(timeseries['time_secs'].values[:100])
    median_dt = np.median(time_diff)
    return round(1.0 / median_dt)


# =============================================================================
# Timing Data Loading
# =============================================================================
def load_start_timings(dataset: str) -> pd.DataFrame:
    """
    Load trial start timings for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name ("sheng2020" or "equalIndifference" or "equalRange")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [trial, time_secs, run_id, sub_id]

    Raises
    ------
    FileNotFoundError
        If the timings file does not exist
    """
    timings_path = get_timings_path(dataset)

    if not timings_path.exists():
        raise FileNotFoundError(
            f"Timings file not found: {timings_path}. "
            f"Run the pupil preprocessing pipeline first."
        )

    return pd.read_csv(timings_path)


def get_subject_run_timings(
    timings_df: pd.DataFrame,
    sub_id: int,
    run_id: int
) -> pd.DataFrame:
    """
    Get trial timings for a specific subject and run.

    Parameters
    ----------
    timings_df : pd.DataFrame
        Full timings dataframe
    sub_id : int
        Subject ID
    run_id : int
        Run ID

    Returns
    -------
    pd.DataFrame
        Filtered timings with columns [trial, time_secs]

    Raises
    ------
    ValueError
        If no timings found for subject/run
    """
    mask = (timings_df['sub_id'] == sub_id) & (timings_df['run_id'] == run_id)
    result = timings_df.loc[mask, ['trial', 'time_secs']].copy()

    if len(result) == 0:
        raise ValueError(f"No timings found for sub-{sub_id}, run-{run_id}")

    return result.reset_index(drop=True)


# =============================================================================
# Subject Discovery
# =============================================================================
def get_available_subjects(dataset: str) -> List[int]:
    """
    Get list of subjects with available timeseries data.

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    List[int]
        Sorted list of subject IDs
    """
    timeseries_dir = get_timeseries_dir(dataset)

    if not timeseries_dir.exists():
        raise FileNotFoundError(f"Timeseries directory not found: {timeseries_dir}")

    files = list(timeseries_dir.glob("sub-*_run-*_timeseries.csv"))
    subjects = set()

    for f in files:
        parts = f.stem.split("_")
        sub_str = parts[0].replace("sub-", "")
        subjects.add(int(sub_str))

    return sorted(list(subjects))

def get_subject_runs(dataset: str, sub_id: int) -> List[int]:
    """
    Get list of available runs for a subject.

    Parameters
    ----------
    dataset : str
        Dataset name
    sub_id : int
        Subject ID

    Returns
    -------
    List[int]
        List of run IDs
    """
    timeseries_dir = get_timeseries_dir(dataset)
    pattern = f"sub-{sub_id:03d}_run-*_timeseries.csv"
    files = list(timeseries_dir.glob(pattern))

    runs = []
    for f in files:
        parts = f.stem.split("_")
        run_str = parts[1].replace("run-", "")
        runs.append(int(run_str))

    return sorted(runs)
