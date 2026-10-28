"""Pupil signal reconstruction and epoch extraction utilities.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from riskhistory.data import (REPO_ROOT, load_pupilbetas, load_behavioral_data,
                              load_behavioral_data_pupilclean, DATASET_RUNS)
# GLM / PRF constants are imported from the fitting script so the
# reconstruction cannot drift from the fitted model
from riskhistory.setup_pupilbetas.run_setup_pupilglm import (
    PRF_MAX_SECS as _PRF_MAX_SECS,
    PRF_WIDTH as _PRF_WIDTH,
    PRF_TIME_TO_PEAK as _PRF_TIME_TO_PEAK,
    NORMALIZE_CONVOLVED as _NORMALIZE_CONVOLVED,
    HIGHPASS_CUTOFF as _HP_CUTOFF,
    FS as _HP_FS,              # samples per second
    ORDER as _HP_ORDER,
)

_EPOCH_FILES = {
    "start":    "pupil_data_alignedStartTime.csv",
    "response": "pupil_data_alignedResponseTime.csv",
}


def load_pupil_epochs(alignment: str = "start") -> pd.DataFrame:
    """Load baseline-corrected pupil epochs merged with behavioral data.

    Applies both pupil-quality exclusion (from the epoch file) and behavioral
    exclusion (trials marked exclude in behavioral_data.csv).

    Parameters
    ----------
    alignment : "start" (stimulus-onset) or "response" (response-locked)

    Returns
    -------
    DataFrame with columns: sub, trial, dataset, timebin, pupil_size,
    plus all behavioral columns (accept, switch, gain, loss, rt, …).
    """
    fname = _EPOCH_FILES[alignment]
    pupil = pd.read_csv(REPO_ROOT / "data" / "preprocessed" / fname,
                        dtype={"exclude": str})

    pupil["pupil_size"]    = pupil["pupil_size"] - pupil["baseline_pupil"]
    # Missing exclusion flags (trials absent from the exclusion table) count as excluded
    pupil["pupil_exclude"] = pupil["exclude"].fillna("true").str.lower() == "true"
    pupil = pupil.drop(columns=["exclude", "baseline_pupil"])
    pupil = pupil[~pupil["pupil_exclude"]].drop(columns=["pupil_exclude"])

    beh = load_behavioral_data(exclude=True)

    return pupil.merge(beh, on=["sub", "trial", "dataset"], how="inner")


def group_timeseries(
    df: pd.DataFrame,
    variable: str,
    dataset: str,
) -> pd.DataFrame:
    """Mean ± SEM pupil timeseries per group of a categorical variable.

    Computes the per-subject mean at each timebin × group, then averages
    across subjects so that the SEM reflects between-subject variability.

    Parameters
    ----------
    df       : output of load_pupil_epochs(), unfiltered by dataset
    variable : binary column to split on (e.g. "switch", "accept")
    dataset  : which dataset to use

    Returns
    -------
    DataFrame with columns: timebin, {variable}, mean, sem
    """
    ds = df[df["dataset"] == dataset].copy()

    sub_mean = (
        ds.groupby(["sub", "timebin", variable])["pupil_size"]
        .mean()
        .reset_index()
    )

    result = (
        sub_mean.groupby(["timebin", variable])["pupil_size"]
        .agg(mean="mean", sem=lambda x: x.sem())
        .reset_index()
    )
    return result


@lru_cache(maxsize=None)
def _cached_pupilbetas() -> pd.DataFrame:
    return load_pupilbetas()


@lru_cache(maxsize=None)
def _cached_timings(alignment: str = "start") -> pd.DataFrame:
    from riskhistory.setup_pupilbetas.utils_get_data import get_all_timings_df
    return get_all_timings_df(alignment)


@lru_cache(maxsize=None)
def _cached_clean_trials(alignment: str = "start") -> pd.DataFrame:
    return load_behavioral_data_pupilclean(alignment)[["sub", "dataset", "trial"]]


def predicted_pupil_sub(
    sub: int,
    dataset: str,
) -> dict[int, dict[str, np.ndarray]]:
    """Reconstruct predicted pupil dilation for all runs of one subject.

    Uses the GLM design matrix and the stored per-trial beta estimates to
    compute the full predicted time-series, then returns it alongside the
    high-pass-filtered observed signal.

    Parameters
    ----------
    sub     : subject ID (integer)
    dataset : one of "equalIndifference", "equalRange", "sheng2020"

    Returns
    -------
    dict keyed by run number (int).  Each value is a dict:
        {
          "predicted"  : np.ndarray, shape (T,),
          "real"       : np.ndarray, shape (T,),
          "time_secs"  : np.ndarray, shape (T,),
        }
    Runs whose pupil timeseries file is missing are skipped with a message.
    Raises ValueError if any regressor of the design matrix has no stored beta.
    """
    from riskhistory.setup_pupilbetas.utils_get_data import get_participant_pupil_data
    from riskhistory.setup_pupilbetas.utils_models import highpass_filter
    from riskhistory.setup_pupilbetas.utils_glm import build_design_matrix_run
    from riskhistory.setup_pupilbetas.utils_prf import get_prf

    betas_df = _cached_pupilbetas()
    start_timings = _cached_timings("start")
    chosen_prf = get_prf(
        max_secs=_PRF_MAX_SECS,
        width=_PRF_WIDTH,
        time_to_peak=_PRF_TIME_TO_PEAK,
    )

    runs = DATASET_RUNS[dataset]
    results: dict[int, dict[str, np.ndarray]] = {}

    sub_timings = start_timings.query("sub == @sub and dataset == @dataset")

    for run in runs:
        try:
            pupil_data = get_participant_pupil_data(sub, run, dataset).copy()
        except FileNotFoundError as exc:
            print(f"  predicted_pupil_sub: skipping sub={sub} run={run} — {exc}")
            continue

        pupil_data["pupil"] = highpass_filter(
            pupil_data["pupil_filtered"].values,
            cutoff=_HP_CUTOFF,
            fs=_HP_FS,
            order=_HP_ORDER,
        )

        design_matrix, betas_name = build_design_matrix_run(
            {run: pupil_data},
            chosen_prf,
            sub_timings,
            sub, run, dataset,
            normalize_convolved=_NORMALIZE_CONVOLVED,
        )

        sub_betas = betas_df.query(
            "sub == @sub and run == @run and dataset == @dataset"
        )
        betas_dict = dict(
            zip(sub_betas["betas_names"], sub_betas["betas_estimates"])
        )
        missing = [name for name in betas_name if name not in betas_dict]
        if missing:
            raise ValueError(
                f"sub={sub} run={run} dataset={dataset}: {len(missing)} of "
                f"{len(betas_name)} regressors have no stored beta "
                f"(e.g. {missing[:3]})"
            )
        beta_vec = np.array([betas_dict[name] for name in betas_name])

        results[run] = {
            "predicted": design_matrix @ beta_vec,
            "real":      pupil_data["pupil"].values,
            "time_secs": pupil_data["time_secs"].values,
        }

    return results


def mean_epochs_sub(
    sub: int,
    dataset: str,
    t_pre: float = 0.0,
    t_post: float = 4.0,
    alignment: str = "start",
) -> dict[str, np.ndarray]:
    """Compute mean ± SEM pupil epochs (predicted and real) across all runs.

    Calls ``predicted_pupil_sub`` to reconstruct the full timeseries (from
    all trials, as in the fit), then epochs around the stimulus-onset or
    button-press timings of the trials that pass behavioral and pupil
    exclusion for that alignment (``load_behavioral_data_pupilclean``) and
    stacks them across runs before
    computing the mean and the across-trial SEM.  No baseline correction is
    applied: both signals are the high-pass-filtered z-scored pupil.

    Parameters
    ----------
    sub     : subject ID (integer)
    dataset : one of "equalIndifference", "equalRange", "sheng2020"
    t_pre     : seconds before each anchor to include
    t_post    : seconds after each anchor to include
    alignment : "start" (stimulus onset) or "response" (button press)

    Returns
    -------
    dict with keys:
        "predicted_mean" : np.ndarray, shape (n_samples,)
        "predicted_sem"  : np.ndarray, shape (n_samples,)
        "real_mean"      : np.ndarray, shape (n_samples,)
        "real_sem"       : np.ndarray, shape (n_samples,)
        "time_axis"      : np.ndarray, shape (n_samples,) — seconds from anchor
    """
    run_data = predicted_pupil_sub(sub, dataset)
    if alignment not in ("start", "response"):
        raise ValueError(f"alignment must be 'start' or 'response', got {alignment!r}")

    clean_trials = _cached_clean_trials(alignment).query(
        "sub == @sub and dataset == @dataset"
    )
    sub_timings = (
        _cached_timings(alignment)
        .query("sub == @sub and dataset == @dataset")
        .merge(clean_trials, on=["sub", "dataset", "trial"], how="inner")
    )

    all_pred: list[np.ndarray] = []
    all_real: list[np.ndarray] = []

    for run, signals in run_data.items():
        anchor_times = sub_timings.query("run == @run")["time_secs"].values
        pred_ep = epochs_data(
            signals["predicted"], signals["time_secs"], anchor_times, t_pre, t_post
        )
        real_ep = epochs_data(
            signals["real"], signals["time_secs"], anchor_times, t_pre, t_post
        )
        if pred_ep.shape[0] > 0:
            all_pred.append(pred_ep)
            all_real.append(real_ep)

    if not all_pred:
        raise ValueError(f"No valid epochs for sub={sub}, dataset={dataset}")

    pred_all = np.vstack(all_pred)   # (total_trials, n_samples)
    real_all = np.vstack(all_real)

    n_samples = pred_all.shape[1]
    n_pre = round(t_pre * _HP_FS)
    time_axis = np.arange(-n_pre, n_samples - n_pre) / _HP_FS

    def _sem(x: np.ndarray) -> np.ndarray:
        if x.shape[0] < 2:
            return np.full(x.shape[1], np.nan)
        return x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])

    return {
        "predicted_mean": pred_all.mean(axis=0),
        "predicted_sem":  _sem(pred_all),
        "real_mean":      real_all.mean(axis=0),
        "real_sem":       _sem(real_all),
        "time_axis":      time_axis,
    }


def epochs_data(
    signal: np.ndarray,
    time_vec: np.ndarray,
    anchor_times: np.ndarray | list[float],
    t_pre: float,
    t_post: float,
) -> np.ndarray:
    """Extract fixed-length epochs from a continuous signal.

    For each anchor time the nearest sample is found, and a window of
    n_pre samples before and n_post samples after that sample is extracted.
    Anchors more than one sample away from the recording, and epochs that
    would require out-of-bounds indexing, are dropped.
    """
    signal   = np.asarray(signal)
    time_vec = np.asarray(time_vec)
    T        = len(signal)
    n_pre, n_post = round(t_pre * _HP_FS), round(t_post * _HP_FS)

    epochs = []
    for t in anchor_times:
        idx = int(np.argmin(np.abs(time_vec - t)))
        if abs(time_vec[idx] - t) > 1 / _HP_FS:
            continue
        i0, i1 = idx - n_pre, idx + n_post
        if i0 >= 0 and i1 <= T:
            epochs.append(signal[i0:i1])

    return np.array(epochs) if epochs else np.empty((0, n_pre + n_post))
