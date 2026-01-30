# =============================================================================
# Pupil Beta Extraction: GLM Utilities
# =============================================================================
# GLM model construction and fitting for trial-by-trial beta extraction.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.signal import convolve
from typing import Tuple, NamedTuple

from .utils_io import (
    load_timeseries,
    get_subject_run_timings,
    get_subject_runs,
    get_analysis_dataset,
    estimate_sampling_rate,
)
from .utils_prf import (
    create_prf_kernel,
    create_dilation_kernel,
    create_constriction_kernel,
    compute_mean_pupil_subject,
    fit_prf,
)


class GLMResult(NamedTuple):
    """Result from GLM fitting."""
    trial_betas: np.ndarray
    trial_ids: np.ndarray
    residuals: np.ndarray
    y_predicted: np.ndarray
    y_observed: np.ndarray
    valid_mask: np.ndarray


# =============================================================================
# Control Regressors
# =============================================================================
def create_control_regressors(timeseries: pd.DataFrame) -> np.ndarray:
    """
    Create control regressors from eye position data.

    Creates: intercept, eye_x, eye_x^2, eye_y, eye_y^2

    Parameters
    ----------
    timeseries : pd.DataFrame
        Timeseries with columns [eye_x, eye_y]

    Returns
    -------
    np.ndarray
        Control regressor matrix (n_timepoints x 5)
    """
    n = len(timeseries)
    eye_x = timeseries['eye_x'].values.astype(float)
    eye_y = timeseries['eye_y'].values.astype(float)

    # Z-score eye positions to avoid numerical issues
    eye_x_mean = np.nanmean(eye_x)
    eye_x_std = np.nanstd(eye_x)
    eye_y_mean = np.nanmean(eye_y)
    eye_y_std = np.nanstd(eye_y)

    # Avoid division by zero
    eye_x_std = eye_x_std if eye_x_std > 0 else 1.0
    eye_y_std = eye_y_std if eye_y_std > 0 else 1.0

    eye_x_z = (eye_x - eye_x_mean) / eye_x_std
    eye_y_z = (eye_y - eye_y_mean) / eye_y_std

    # Create control matrix: [intercept, eye_x, eye_x^2, eye_y, eye_y^2]
    controls = np.column_stack([
        np.ones(n),        # intercept
        eye_x_z,           # eye_x (z-scored)
        eye_x_z ** 2,      # eye_x^2
        eye_y_z,           # eye_y (z-scored)
        eye_y_z ** 2,      # eye_y^2
    ])

    return controls


# =============================================================================
# Design Matrix Construction
# =============================================================================
def create_trial_regressors(
    onset_times: pd.DataFrame,
    n_timepoints: int,
    dilation_kernel: np.ndarray,
    sampling_rate: float,
    run_start_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create trial regressors (dilation-convolved delta functions).

    Each trial gets its own regressor with the dilation component of the PRF.

    Parameters
    ----------
    onset_times : pd.DataFrame
        Trial onset times with columns [trial, time_secs]
    n_timepoints : int
        Number of timepoints in the run
    dilation_kernel : np.ndarray
        Dilation-only kernel for convolution
    sampling_rate : float
        Sampling rate in Hz
    run_start_time : float
        Time of first sample in seconds

    Returns
    -------
    X_trials : np.ndarray
        Trial regressor matrix (n_timepoints x n_trials)
    trial_ids : np.ndarray
        Trial IDs corresponding to each column
    """
    n_trials = len(onset_times)
    X_trials = np.zeros((n_timepoints, n_trials))
    trial_ids = onset_times['trial'].values.astype(int)

    for i, (_, row) in enumerate(onset_times.iterrows()):
        onset_sec = row['time_secs']
        onset_sample = int((onset_sec - run_start_time) * sampling_rate)

        if onset_sample < 0 or onset_sample >= n_timepoints:
            continue

        delta = np.zeros(n_timepoints)
        delta[onset_sample] = 1.0

        convolved = convolve(delta, dilation_kernel, mode='full')[:n_timepoints]
        X_trials[:, i] = convolved

    return X_trials, trial_ids


def create_constriction_regressor(
    onset_times: pd.DataFrame,
    n_timepoints: int,
    constriction_kernel: np.ndarray,
    sampling_rate: float,
    run_start_time: float
) -> np.ndarray:
    """
    Create shared constriction regressor (sum of all trial deltas convolved with constriction).

    All trials share a single constriction amplitude fit.

    Parameters
    ----------
    onset_times : pd.DataFrame
        Trial onset times with columns [trial, time_secs]
    n_timepoints : int
        Number of timepoints in the run
    constriction_kernel : np.ndarray
        Constriction-only kernel for convolution
    sampling_rate : float
        Sampling rate in Hz
    run_start_time : float
        Time of first sample in seconds

    Returns
    -------
    np.ndarray
        Single constriction regressor (n_timepoints,)
    """
    # Create sum of all trial deltas
    delta_sum = np.zeros(n_timepoints)

    for _, row in onset_times.iterrows():
        onset_sec = row['time_secs']
        onset_sample = int((onset_sec - run_start_time) * sampling_rate)

        if onset_sample < 0 or onset_sample >= n_timepoints:
            continue

        delta_sum[onset_sample] = 1.0

    # Convolve with constriction kernel
    constriction_regressor = convolve(delta_sum, constriction_kernel, mode='full')[:n_timepoints]

    return constriction_regressor


# =============================================================================
# GLM Fitting
# =============================================================================
def fit_glm_with_controls(
    y: np.ndarray,
    X_trials: np.ndarray,
    X_constriction: np.ndarray,
    X_controls: np.ndarray
) -> GLMResult:
    """
    Fit GLM with trial regressors, constriction regressor, and control regressors.

    The model is:
    y = X_controls @ beta_controls + X_constriction * beta_constriction + X_trials @ beta_trials + error

    The dilation (per-trial) and constriction (shared) are fitted separately.

    Parameters
    ----------
    y : np.ndarray
        Response variable (n_timepoints,)
    X_trials : np.ndarray
        Trial regressor matrix (n_timepoints x n_trials) - dilation convolved
    X_constriction : np.ndarray
        Shared constriction regressor (n_timepoints,)
    X_controls : np.ndarray
        Control regressor matrix (n_timepoints x n_controls)

    Returns
    -------
    GLMResult
        Named tuple with trial_betas, residuals, predictions, etc.

    Raises
    ------
    ValueError
        If insufficient valid data points
    """
    # Handle NaN in y and controls
    valid_mask = ~np.isnan(y)
    for col in range(X_controls.shape[1]):
        valid_mask &= ~np.isnan(X_controls[:, col])

    y_valid = y[valid_mask]
    X_trials_valid = X_trials[valid_mask, :]
    X_constriction_valid = X_constriction[valid_mask].reshape(-1, 1)
    X_controls_valid = X_controls[valid_mask, :]

    # +1 for constriction regressor
    n_regressors = X_trials.shape[1] + 1 + X_controls.shape[1]
    if len(y_valid) < n_regressors + 1:
        raise ValueError(
            f"Insufficient data points ({len(y_valid)}) for "
            f"{n_regressors} regressors"
        )

    # Combine: controls first, then constriction, then trials
    X_full = np.column_stack([X_controls_valid, X_constriction_valid, X_trials_valid])

    # Solve using least squares
    betas, _, _, _ = np.linalg.lstsq(X_full, y_valid, rcond=None)

    # Extract trial betas (skip control betas and constriction beta)
    n_controls = X_controls.shape[1]
    trial_betas = betas[n_controls + 1:]  # +1 for constriction

    # Compute predictions and residuals
    y_predicted = X_full @ betas
    residuals = y_valid - y_predicted

    # Store observed values
    y_observed = y_valid

    return GLMResult(
        trial_betas=trial_betas,
        trial_ids=np.arange(X_trials.shape[1]),
        residuals=residuals,
        y_predicted=y_predicted,
        y_observed=y_observed,
        valid_mask=valid_mask
    )


def fit_glm_run(
    timeseries: pd.DataFrame,
    onset_times: pd.DataFrame,
    prf_params: Tuple[float, float, float, float, float, float],
    prf_duration: float = 10.0
) -> Tuple[pd.DataFrame, GLMResult]:
    """
    Fit GLM to extract trial-by-trial pupil betas for a single run.

    The GLM uses:
    - Per-trial regressors: dilation component convolved with trial delta
    - Shared constriction regressor: constriction component convolved with all trial deltas

    Parameters
    ----------
    timeseries : pd.DataFrame
        Timeseries with columns [time_secs, pupil_filtered, eye_x, eye_y]
    onset_times : pd.DataFrame
        Trial onset times with columns [trial, time_secs]
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d) PRF parameters
    prf_duration : float
        Duration of PRF kernel in seconds

    Returns
    -------
    betas_df : pd.DataFrame
        DataFrame with columns [trial, pupil_beta]
    glm_result : GLMResult
        Full GLM results including residuals
    """
    sampling_rate = estimate_sampling_rate(timeseries)

    time_vec = timeseries['time_secs'].values
    pupil_vec = timeseries['pupil_filtered'].values
    n_timepoints = len(pupil_vec)
    run_start_time = time_vec[0]

    # Create separate dilation and constriction kernels
    dilation_kernel = create_dilation_kernel(prf_params, prf_duration, sampling_rate)
    constriction_kernel = create_constriction_kernel(prf_params, prf_duration, sampling_rate)

    # Create trial regressors (dilation only)
    X_trials, trial_ids = create_trial_regressors(
        onset_times, n_timepoints, dilation_kernel, sampling_rate, run_start_time
    )

    # Create shared constriction regressor
    X_constriction = create_constriction_regressor(
        onset_times, n_timepoints, constriction_kernel, sampling_rate, run_start_time
    )

    # Create control regressors
    X_controls = create_control_regressors(timeseries)

    # Fit GLM
    glm_result = fit_glm_with_controls(pupil_vec, X_trials, X_constriction, X_controls)

    betas_df = pd.DataFrame({
        'trial': onset_times['trial'].values.astype(int),
        'pupil_beta': glm_result.trial_betas
    })

    return betas_df, glm_result


# =============================================================================
# Subject-Level Extraction
# =============================================================================
def extract_betas_subject(
    sub_id: int,
    source_dataset: str,
    timings_df: pd.DataFrame,
    prf_params: Tuple[float, float, float, float, float, float],
    prf_duration: float = 10.0
) -> pd.DataFrame:
    """
    Extract pupil betas for all runs of a subject.

    Parameters
    ----------
    sub_id : int
        Subject ID
    source_dataset : str
        Source dataset name ("sheng2020" or "narps")
    timings_df : pd.DataFrame
        Full timings dataframe
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)
    prf_duration : float
        Duration of PRF kernel in seconds

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [sub, dataset, trial, run, pupil_beta]
    """
    runs = get_subject_runs(source_dataset, sub_id)
    analysis_dataset = get_analysis_dataset(source_dataset, sub_id)
    all_betas = []

    for run_id in runs:
        timeseries = load_timeseries(sub_id, run_id, source_dataset)
        onset_times = get_subject_run_timings(timings_df, sub_id, run_id)

        betas_df, _ = fit_glm_run(timeseries, onset_times, prf_params, prf_duration)
        betas_df['run'] = run_id
        betas_df['sub'] = sub_id
        betas_df['dataset'] = analysis_dataset

        all_betas.append(betas_df)

    result = pd.concat(all_betas, ignore_index=True)
    return result[['sub', 'dataset', 'trial', 'run', 'pupil_beta']]


def extract_betas_with_prf(
    sub_id: int,
    source_dataset: str,
    timings_df: pd.DataFrame,
    prf_params: Tuple[float, float, float, float, float, float],
    prf_duration: float = 10.0,
    return_diagnostics: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Extract trial betas using pre-fitted PRF parameters.

    Parameters
    ----------
    sub_id : int
        Subject ID
    source_dataset : str
        Source dataset name ("sheng2020" or "narps")
    timings_df : pd.DataFrame
        Full timings dataframe
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)
    prf_duration : float
        Duration of PRF kernel for GLM (seconds)
    return_diagnostics : bool
        If True, include diagnostic data in prf_info

    Returns
    -------
    betas_df : pd.DataFrame
        Trial betas with columns [sub, dataset, trial, run, pupil_beta]
    prf_info : dict
        PRF parameters and fitting info
    """
    analysis_dataset = get_analysis_dataset(source_dataset, sub_id)
    amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d = prf_params

    prf_info = {
        'sub': sub_id,
        'dataset': analysis_dataset,
        'source_dataset': source_dataset,
        'amplitude': amplitude,
        'a2': a2,
        'tau_rise_c': tau_rise_c,
        'tau_decay_c': tau_decay_c,
        'tau_rise_d': tau_rise_d,
        'tau_decay_d': tau_decay_d,
        'prf_mode': 'dataset',
    }

    # Extract betas
    runs = get_subject_runs(source_dataset, sub_id)
    all_betas = []
    all_residuals = []

    for run_id in runs:
        timeseries = load_timeseries(sub_id, run_id, source_dataset)
        onset_times = get_subject_run_timings(timings_df, sub_id, run_id)

        betas_df, glm_result = fit_glm_run(
            timeseries, onset_times, prf_params, prf_duration
        )
        betas_df['run'] = run_id
        betas_df['sub'] = sub_id
        betas_df['dataset'] = analysis_dataset

        all_betas.append(betas_df)

        if return_diagnostics:
            all_residuals.append({
                'run': run_id,
                'residuals': glm_result.residuals,
                'y_predicted': glm_result.y_predicted,
                'y_observed': glm_result.y_observed
            })

    if return_diagnostics:
        prf_info['run_diagnostics'] = all_residuals

    result = pd.concat(all_betas, ignore_index=True)
    result = result[['sub', 'dataset', 'trial', 'run', 'pupil_beta']]

    return result, prf_info


def fit_prf_and_extract_betas(
    sub_id: int,
    source_dataset: str,
    timings_df: pd.DataFrame,
    epoch_start: float = 0.0,
    epoch_end: float = 5.0,
    prf_duration: float = 10.0,
    return_diagnostics: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Fit subject-specific PRF and extract trial betas.

    Parameters
    ----------
    sub_id : int
        Subject ID
    source_dataset : str
        Source dataset name ("sheng2020" or "narps")
    timings_df : pd.DataFrame
        Full timings dataframe
    epoch_start : float
        Start of epoch for PRF fitting (seconds)
    epoch_end : float
        End of epoch for PRF fitting (seconds)
    prf_duration : float
        Duration of PRF kernel for GLM (seconds)
    return_diagnostics : bool
        If True, include diagnostic data in prf_info

    Returns
    -------
    betas_df : pd.DataFrame
        Trial betas with columns [sub, dataset, trial, run, pupil_beta]
    prf_info : dict
        PRF parameters and fitting info (includes diagnostics if requested)
    """
    analysis_dataset = get_analysis_dataset(source_dataset, sub_id)

    # Compute mean pupil response
    time_vec, mean_pupil, n_epochs = compute_mean_pupil_subject(
        sub_id, source_dataset, timings_df, epoch_start, epoch_end
    )

    # Fit PRF (6 parameters)
    amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d = fit_prf(time_vec, mean_pupil)
    prf_params = (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)

    prf_info = {
        'sub': sub_id,
        'dataset': analysis_dataset,
        'source_dataset': source_dataset,
        'amplitude': amplitude,
        'a2': a2,
        'tau_rise_c': tau_rise_c,
        'tau_decay_c': tau_decay_c,
        'tau_rise_d': tau_rise_d,
        'tau_decay_d': tau_decay_d,
        'n_epochs_for_fit': n_epochs
    }

    if return_diagnostics:
        from .utils_prf import biphasic_prf
        prf_info['time_vec'] = time_vec
        prf_info['mean_pupil'] = mean_pupil
        prf_info['fitted_prf'] = biphasic_prf(
            time_vec, amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d
        )

    # Extract betas with diagnostics
    runs = get_subject_runs(source_dataset, sub_id)
    all_betas = []
    all_residuals = []

    for run_id in runs:
        timeseries = load_timeseries(sub_id, run_id, source_dataset)
        onset_times = get_subject_run_timings(timings_df, sub_id, run_id)

        betas_df, glm_result = fit_glm_run(
            timeseries, onset_times, prf_params, prf_duration
        )
        betas_df['run'] = run_id
        betas_df['sub'] = sub_id
        betas_df['dataset'] = analysis_dataset

        all_betas.append(betas_df)

        if return_diagnostics:
            all_residuals.append({
                'run': run_id,
                'residuals': glm_result.residuals,
                'y_predicted': glm_result.y_predicted,
                'y_observed': glm_result.y_observed
            })

    if return_diagnostics:
        prf_info['run_diagnostics'] = all_residuals

    result = pd.concat(all_betas, ignore_index=True)
    result = result[['sub', 'dataset', 'trial', 'run', 'pupil_beta']]

    return result, prf_info
