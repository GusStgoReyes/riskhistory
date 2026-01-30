# =============================================================================
# Pupil Beta Extraction: PRF Utilities
# =============================================================================
# Pupil Response Function (PRF) definitions and fitting functions.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Optional, List

from .utils_io import (
    load_timeseries,
    get_subject_run_timings,
    get_subject_runs,
    estimate_sampling_rate,
)


# =============================================================================
# PRF Functions
# =============================================================================
def biphasic_prf(
    t,
    amplitude,
    a2=0.3,
    tau_rise_c=0.2,
    tau_decay_c=0.7,
    tau_rise_d=0.8,
    tau_decay_d=3.5,
):
    """
    Biphasic pupil response function with separate constriction and dilation.

    Parameters
    ----------
    t : array-like
        Time vector in seconds (must be >= 0)
    amplitude : float
        Amplitude of the dilation component
    a2 : float
        Amplitude of the constriction component
    tau_rise_c : float
        Rise time constant for constriction (seconds)
    tau_decay_c : float
        Decay time constant for constriction (seconds)
    tau_rise_d : float
        Rise time constant for dilation (seconds)
    tau_decay_d : float
        Decay time constant for dilation (seconds)

    Returns
    -------
    np.ndarray
        PRF values at each time point
    """
    t = np.asarray(t)
    t = np.maximum(t, 0)

    constriction = -a2 * (
        np.exp(-t / tau_decay_c)
        - np.exp(-t / tau_rise_c)
    )

    dilation = amplitude * (
        np.exp(-t / tau_decay_d)
        - np.exp(-t / tau_rise_d)
    )

    return constriction + dilation


def dilation_component(
    t,
    amplitude=1.0,
    tau_rise_d=0.8,
    tau_decay_d=3.5,
):
    """
    Dilation component of the PRF (for per-trial convolution).

    Parameters
    ----------
    t : array-like
        Time vector in seconds
    amplitude : float
        Amplitude of dilation
    tau_rise_d : float
        Rise time constant for dilation (seconds)
    tau_decay_d : float
        Decay time constant for dilation (seconds)

    Returns
    -------
    np.ndarray
        Dilation component values
    """
    t = np.asarray(t)
    t = np.maximum(t, 0)
    return amplitude * (np.exp(-t / tau_decay_d) - np.exp(-t / tau_rise_d))


def constriction_component(
    t,
    a2=0.3,
    tau_rise_c=0.2,
    tau_decay_c=0.7,
):
    """
    Constriction component of the PRF (shared across all trials).

    Parameters
    ----------
    t : array-like
        Time vector in seconds
    a2 : float
        Amplitude of constriction
    tau_rise_c : float
        Rise time constant for constriction (seconds)
    tau_decay_c : float
        Decay time constant for constriction (seconds)

    Returns
    -------
    np.ndarray
        Constriction component values
    """
    t = np.asarray(t)
    t = np.maximum(t, 0)
    return -a2 * (np.exp(-t / tau_decay_c) - np.exp(-t / tau_rise_c))


def _biphasic_prf_fittable(
    t: np.ndarray,
    amplitude: float,
    a2: float,
    tau_rise_c: float,
    tau_decay_c: float,
    tau_rise_d: float,
    tau_decay_d: float,
) -> np.ndarray:
    """Biphasic PRF with all 6 parameters for curve_fit."""
    return biphasic_prf(t, amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)


# =============================================================================
# Epoch Extraction
# =============================================================================
def extract_epoch(
    timeseries: pd.DataFrame,
    onset_time: float,
    epoch_start: float,
    epoch_end: float,
    sampling_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a single epoch from timeseries.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Timeseries with columns [time_secs, pupil_filtered]
    onset_time : float
        Event onset time in seconds
    epoch_start : float
        Start of epoch relative to onset (seconds)
    epoch_end : float
        End of epoch relative to onset (seconds)
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    time_relative : np.ndarray
        Time relative to onset (always n_samples length)
    pupil_values : np.ndarray
        Pupil values (always n_samples length, padded with NaN if needed)
    """
    n_samples = int((epoch_end - epoch_start) * sampling_rate)
    time_relative = np.linspace(epoch_start, epoch_end, n_samples)

    time_vec = timeseries['time_secs'].values
    pupil_vec = timeseries['pupil_filtered'].values

    # Find samples within epoch window
    epoch_mask = (time_vec >= onset_time + epoch_start) & (time_vec < onset_time + epoch_end)

    if epoch_mask.sum() < n_samples * 0.5:
        return time_relative, np.full(n_samples, np.nan)

    epoch_pupil = pupil_vec[epoch_mask]

    # Ensure output is always n_samples length
    if len(epoch_pupil) < n_samples:
        # Pad with NaN
        padded = np.full(n_samples, np.nan)
        padded[:len(epoch_pupil)] = epoch_pupil
        epoch_pupil = padded
    elif len(epoch_pupil) > n_samples:
        # Truncate
        epoch_pupil = epoch_pupil[:n_samples]

    return time_relative, epoch_pupil

def compute_mean_pupil_run(
    timeseries: pd.DataFrame,
    onset_times: pd.DataFrame,
    epoch_start: float = 0.0,
    epoch_end: float = 5.0,
    baseline_end: float = 0.5,
    sampling_rate: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute mean baseline-corrected pupil dilation for a single run.

    Parameters
    ----------
    timeseries : pd.DataFrame
        Timeseries with columns [time_secs, pupil_filtered]
    onset_times : pd.DataFrame
        Trial onset times with columns [trial, time_secs]
    epoch_start : float
        Start of epoch relative to trial onset (seconds)
    epoch_end : float
        End of epoch relative to trial onset (seconds)
    baseline_end : float
        End of baseline period (seconds from epoch_start)
    sampling_rate : float, optional
        Sampling rate. If None, estimated from data.

    Returns
    -------
    time_relative : np.ndarray
        Time vector relative to trial onset
    mean_pupil : np.ndarray
        Mean baseline-corrected pupil dilation
    n_valid : int
        Number of valid epochs included
    """
    if sampling_rate is None:
        sampling_rate = estimate_sampling_rate(timeseries)

    n_samples = int((epoch_end - epoch_start) * sampling_rate)
    time_relative = np.linspace(epoch_start, epoch_end, n_samples)

    epochs = []
    for _, row in onset_times.iterrows():
        _, pupil = extract_epoch(
            timeseries, row['time_secs'], epoch_start, epoch_end, sampling_rate
        )
        pupil = pupil - pupil[0]

        if np.isnan(pupil).all():
            continue
        epochs.append(pupil)

    if len(epochs) == 0:
        return time_relative, np.full(n_samples, np.nan), 0

    epochs_array = np.array(epochs)
    mean_pupil = np.nanmean(epochs_array, axis=0)

    return time_relative, mean_pupil, len(epochs)

def compute_mean_pupil_subject(
    sub_id: int,
    dataset: str,
    timings_df: pd.DataFrame,
    epoch_start: float = 0.0,
    epoch_end: float = 5.0,
    baseline_end: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute mean pupil dilation across all runs for a subject.

    Parameters
    ----------
    sub_id : int
        Subject ID
    dataset : str
        Dataset name
    timings_df : pd.DataFrame
        Full timings dataframe
    epoch_start : float
        Start of epoch relative to trial onset (seconds)
    epoch_end : float
        End of epoch relative to trial onset (seconds)
    baseline_end : float
        End of baseline period (seconds)

    Returns
    -------
    time_relative : np.ndarray
        Time vector relative to trial onset
    mean_pupil : np.ndarray
        Mean pupil dilation across all runs
    total_epochs : int
        Total number of valid epochs
    """
    runs = get_subject_runs(dataset, sub_id)

    all_mean_pupils = []
    total_epochs = 0
    time_relative = None
    sampling_rate = None

    for run_id in runs:
        timeseries = load_timeseries(sub_id, run_id, dataset)
        onset_times = get_subject_run_timings(timings_df, sub_id, run_id)

        if sampling_rate is None:
            sampling_rate = estimate_sampling_rate(timeseries)

        time_rel, mean_pup, n_valid = compute_mean_pupil_run(
            timeseries, onset_times, epoch_start, epoch_end, baseline_end, sampling_rate
        )

        if time_relative is None:
            time_relative = time_rel

        if n_valid > 0:
            all_mean_pupils.append(mean_pup)
            total_epochs += n_valid

    if len(all_mean_pupils) == 0:
        n_samples = int((epoch_end - epoch_start) * (sampling_rate or 125))
        time_relative = np.linspace(epoch_start, epoch_end, n_samples)
        return time_relative, np.full(n_samples, np.nan), 0

    mean_pupil = np.nanmean(np.array(all_mean_pupils), axis=0)
    return time_relative, mean_pupil, total_epochs


# =============================================================================
# PRF Fitting
# =============================================================================
def fit_prf(
    time_vec: np.ndarray,
    mean_pupil: np.ndarray,
    initial_params: Optional[dict] = None
) -> Tuple[float, float, float, float, float, float]:
    """
    Fit biphasic PRF parameters to mean pupil response.

    Parameters
    ----------
    time_vec : np.ndarray
        Time vector in seconds
    mean_pupil : np.ndarray
        Mean pupil dilation values
    initial_params : dict, optional
        Initial parameter guesses with keys:
        amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d

    Returns
    -------
    tuple of 6 floats
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)

    Raises
    ------
    ValueError
        If fitting fails or insufficient data
    """
    if initial_params is None:
        initial_params = {
            'amplitude': 1.0,
            'a2': 0.3,
            'tau_rise_c': 0.2,
            'tau_decay_c': 0.7,
            'tau_rise_d': 0.8,
            'tau_decay_d': 3.5,
        }

    # Remove NaN values
    valid_mask = ~np.isnan(mean_pupil)
    time_valid = time_vec[valid_mask]
    pupil_valid = mean_pupil[valid_mask]

    if len(pupil_valid) < 10:
        raise ValueError(
            f"Insufficient valid data points for PRF fitting: {len(pupil_valid)}"
        )

    # Parameter bounds: [amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d]
    bounds = (
        [0.01, 0.01, 0.05, 0.1, 0.1, 0.5],   # Lower bounds
        [30.0, 30.0, 2.0, 5.0, 5.0, 10.0]    # Upper bounds
    )

    p0 = [
        initial_params['amplitude'],
        initial_params['a2'],
        initial_params['tau_rise_c'],
        initial_params['tau_decay_c'],
        initial_params['tau_rise_d'],
        initial_params['tau_decay_d'],
    ]

    popt, _ = curve_fit(
        _biphasic_prf_fittable,
        time_valid,
        pupil_valid,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    return tuple(popt)


def create_prf_kernel(
    prf_params: Tuple[float, float, float, float, float, float],
    duration: float,
    sampling_rate: float
) -> np.ndarray:
    """
    Create the full PRF kernel for convolution (both components).

    Parameters
    ----------
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)
    duration : float
        Duration of the kernel in seconds
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    np.ndarray
        Full PRF kernel (constriction + dilation)
    """
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate

    amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d = prf_params
    kernel = biphasic_prf(t, amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)

    return kernel


def create_dilation_kernel(
    prf_params: Tuple[float, float, float, float, float, float],
    duration: float,
    sampling_rate: float
) -> np.ndarray:
    """
    Create dilation-only kernel for per-trial convolution.

    Parameters
    ----------
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)
    duration : float
        Duration of the kernel in seconds
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    np.ndarray
        Dilation-only kernel
    """
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate

    amplitude, _, _, _, tau_rise_d, tau_decay_d = prf_params
    kernel = dilation_component(t, amplitude, tau_rise_d, tau_decay_d)

    return kernel


def create_constriction_kernel(
    prf_params: Tuple[float, float, float, float, float, float],
    duration: float,
    sampling_rate: float
) -> np.ndarray:
    """
    Create constriction-only kernel for shared trial regressor.

    Parameters
    ----------
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)
    duration : float
        Duration of the kernel in seconds
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    np.ndarray
        Constriction-only kernel
    """
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate

    _, a2, tau_rise_c, tau_decay_c, _, _ = prf_params
    kernel = constriction_component(t, a2, tau_rise_c, tau_decay_c)

    return kernel


# =============================================================================
# Dataset-Level PRF Estimation
# =============================================================================
def compute_mean_pupil_dataset(
    source_dataset: str,
    timings_df: pd.DataFrame,
    subjects: List[int],
    epoch_start: float = 0.0,
    epoch_end: float = 5.0,
    baseline_end: float = 0.5,
    common_sampling_rate: float = 125.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute grand average pupil response across all subjects in a dataset.

    Parameters
    ----------
    source_dataset : str
        Source dataset name ("sheng2020" or "narps")
    timings_df : pd.DataFrame
        Full timings dataframe
    subjects : list
        List of subject IDs to include
    epoch_start : float
        Start of epoch relative to trial onset (seconds)
    epoch_end : float
        End of epoch relative to trial onset (seconds)
    baseline_end : float
        End of baseline period (seconds)
    common_sampling_rate : float
        Common sampling rate to interpolate all subjects to (Hz)

    Returns
    -------
    time_relative : np.ndarray
        Time vector relative to trial onset
    mean_pupil : np.ndarray
        Grand average pupil dilation across all subjects
    total_epochs : int
        Total number of valid epochs across all subjects
    """
    # Create common time grid
    n_samples = int((epoch_end - epoch_start) * common_sampling_rate)
    time_relative = np.linspace(epoch_start, epoch_end, n_samples)

    all_subject_means = []
    total_epochs = 0

    for sub_id in subjects:
        time_rel, mean_pup, n_epochs = compute_mean_pupil_subject(
            sub_id, source_dataset, timings_df,
            epoch_start, epoch_end, baseline_end
        )

        if n_epochs > 0 and not np.isnan(mean_pup).all():
            # Interpolate to common grid if different sampling rate
            if len(mean_pup) != n_samples:
                mean_pup_interp = np.interp(time_relative, time_rel, mean_pup)
                all_subject_means.append(mean_pup_interp)
            else:
                all_subject_means.append(mean_pup)
            total_epochs += n_epochs

    if len(all_subject_means) == 0:
        return time_relative, np.full(n_samples, np.nan), 0

    # Grand average across subjects
    grand_mean = np.nanmean(np.array(all_subject_means), axis=0)

    return time_relative, grand_mean, total_epochs


def fit_dataset_prf(
    source_dataset: str,
    timings_df: pd.DataFrame,
    subjects: List[int],
    epoch_start: float = 0.0,
    epoch_end: float = 5.0,
    baseline_end: float = 0.5
) -> Tuple[Tuple[float, float, float, float, float, float], dict]:
    """
    Fit PRF parameters on the dataset grand average.

    Parameters
    ----------
    source_dataset : str
        Source dataset name ("sheng2020" or "narps")
    timings_df : pd.DataFrame
        Full timings dataframe
    subjects : list
        List of subject IDs to include
    epoch_start : float
        Start of epoch relative to trial onset (seconds)
    epoch_end : float
        End of epoch relative to trial onset (seconds)
    baseline_end : float
        End of baseline period (seconds)

    Returns
    -------
    prf_params : tuple
        (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)
    prf_info : dict
        Dictionary with fitting information including:
        - time_vec, mean_pupil, fitted_prf for plotting
        - n_subjects, total_epochs
    """
    print(f"    Computing grand average pupil response...")
    time_vec, mean_pupil, total_epochs = compute_mean_pupil_dataset(
        source_dataset, timings_df, subjects,
        epoch_start, epoch_end, baseline_end
    )

    print(f"    Fitting PRF on {len(subjects)} subjects ({total_epochs} total epochs)...")
    amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d = fit_prf(time_vec, mean_pupil)
    prf_params = (amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d)

    prf_info = {
        'source_dataset': source_dataset,
        'n_subjects': len(subjects),
        'total_epochs': total_epochs,
        'amplitude': amplitude,
        'a2': a2,
        'tau_rise_c': tau_rise_c,
        'tau_decay_c': tau_decay_c,
        'tau_rise_d': tau_rise_d,
        'tau_decay_d': tau_decay_d,
        'time_vec': time_vec,
        'mean_pupil': mean_pupil,
        'fitted_prf': biphasic_prf(
            time_vec, amplitude, a2, tau_rise_c, tau_decay_c, tau_rise_d, tau_decay_d
        ),
    }

    print(f"    PRF params: amp={amplitude:.3f}, a2={a2:.3f}, "
          f"tau_rise_c={tau_rise_c:.3f}, tau_decay_c={tau_decay_c:.3f}, "
          f"tau_rise_d={tau_rise_d:.3f}, tau_decay_d={tau_decay_d:.3f}")

    return prf_params, prf_info
