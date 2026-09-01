"""Pupil signal reconstruction and epoch extraction utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskhistory.data import REPO_ROOT, load_pupilbetas, load_behavioral_data, DATASET_RUNS

# ---------------------------------------------------------------------------
# GLM / PRF constants — must match those used during model fitting
# ---------------------------------------------------------------------------
_PRF_MAX_SECS     = 4
_PRF_WIDTH        = 10.1
_PRF_TIME_TO_PEAK = 930
_HP_CUTOFF        = 0.05
_HP_FS            = 25       # samples per second
_HP_ORDER         = 3

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
    pupil["pupil_exclude"] = pupil["exclude"].astype(str).str.lower() == "true"
    pupil = pupil.drop(columns=["exclude", "baseline_pupil"])
    pupil = pupil[~pupil["pupil_exclude"]].drop(columns=["pupil_exclude"])

    beh = load_behavioral_data(exclude=True)

    return pupil.merge(beh, on=["sub", "trial", "dataset"], how="inner")


def add_choice_alteration(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `choice_alteration` (Low/Medium/High) column, mirroring the
    formula in analysis_pupilbetas/0_config.R: how much the current choice
    deviates from the average of the previous 2 choices.

    Parameters
    ----------
    df : DataFrame with `accept` (0/1), `choice_prev1`, `choice_prev2`
        (-1/1, NaN if unavailable)

    Returns
    -------
    df with an added `choice_alteration` categorical column (NaN where
    choice_prev1/choice_prev2 are unavailable).
    """
    df = df.copy()
    choice_prev1_01 = (df["choice_prev1"] + 1) / 2
    choice_prev2_01 = (df["choice_prev2"] + 1) / 2
    choice_alteration_num = 2 * (
        (df["accept"] - (choice_prev1_01 + choice_prev2_01) / 2).abs() - 0.5
    )

    df["choice_alteration"] = pd.cut(
        choice_alteration_num,
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return df


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
    Runs that fail (missing data, etc.) are silently skipped.
    """
    from riskhistory.setup_pupilbetas.utils_get_data import get_participant_pupil_data
    from riskhistory.setup_pupilbetas.utils_get_data import get_all_timings_df
    from riskhistory.setup_pupilbetas.utils_models import highpass_filter
    from riskhistory.setup_pupilbetas.utils_glm import build_design_matrix_run
    from riskhistory.setup_pupilbetas.utils_prf import get_prf

    betas_df = load_pupilbetas()
    start_timings = get_all_timings_df("start")
    chosen_prf = get_prf(
        max_secs=_PRF_MAX_SECS,
        width=_PRF_WIDTH,
        time_to_peak=_PRF_TIME_TO_PEAK,
    )

    runs = DATASET_RUNS[dataset]
    results: dict[int, dict[str, np.ndarray]] = {}

    for run in runs:
        try:
            pupil_data = get_participant_pupil_data(sub, run, dataset).copy()
            pupil_data["pupil"] = highpass_filter(
                pupil_data["pupil_filtered"].values,
                cutoff=_HP_CUTOFF,
                fs=_HP_FS,
                order=_HP_ORDER,
            )

            run_timings = start_timings.query(
                "sub == @sub and dataset == @dataset"
            )

            design_matrix, betas_name = build_design_matrix_run(
                {run: pupil_data},
                chosen_prf,
                run_timings,
                sub, run, dataset,
            )

            sub_betas = betas_df.query(
                "sub == @sub and run == @run and dataset == @dataset"
            )
            betas_dict = dict(
                zip(sub_betas["betas_names"], sub_betas["betas_estimates"])
            )
            beta_vec = np.array(
                [betas_dict.get(name, 0.0) for name in betas_name]
            )

            results[run] = {
                "predicted": design_matrix @ beta_vec,
                "real":      pupil_data["pupil"].values,
                "time_secs": pupil_data["time_secs"].values,
            }

        except Exception as exc:
            print(f"  predicted_pupil_sub: skipping sub={sub} run={run} — {exc}")

    return results


def mean_epochs_sub(
    sub: int,
    dataset: str,
    t_pre: float = 0.0,
    t_post: float = 4.0,
) -> dict[str, np.ndarray]:
    """Compute mean ± SEM pupil epochs (predicted and real) across all runs.

    Calls ``predicted_pupil_sub`` to reconstruct the full timeseries, then
    epochs around start timings with ``epochs_data`` and stacks trials from
    all runs before computing the mean and SEM.

    Parameters
    ----------
    sub     : subject ID (integer)
    dataset : one of "equalIndifference", "equalRange", "sheng2020"
    t_pre   : seconds before each start timing to include
    t_post  : seconds after each start timing to include

    Returns
    -------
    dict with keys:
        "predicted_mean" : np.ndarray, shape (n_samples,)
        "predicted_sem"  : np.ndarray, shape (n_samples,)
        "real_mean"      : np.ndarray, shape (n_samples,)
        "real_sem"       : np.ndarray, shape (n_samples,)
        "time_axis"      : np.ndarray, shape (n_samples,) — seconds from anchor
    """
    from riskhistory.setup_pupilbetas.utils_get_data import get_all_timings_df

    run_data = predicted_pupil_sub(sub, dataset)
    start_timings = get_all_timings_df("start")
    sub_timings = start_timings.query("sub == @sub and dataset == @dataset")

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
    n_pre = int(t_pre * _HP_FS)
    time_axis = np.arange(-n_pre, n_samples - n_pre) / _HP_FS

    n = pred_all.shape[0]
    return {
        "predicted_mean": pred_all.mean(axis=0),
        "predicted_sem":  pred_all.std(axis=0, ddof=1) / np.sqrt(n),
        "real_mean":      real_all.mean(axis=0),
        "real_sem":       real_all.std(axis=0, ddof=1) / np.sqrt(n),
        "time_axis":      time_axis,
    }


def epochs_data(
    signal: np.ndarray,
    time_vec: np.ndarray,
    anchor_times: np.ndarray | list[float],
    t_pre: int,
    t_post: int,
) -> np.ndarray:
    """Extract fixed-length epochs from a continuous signal.

    For each anchor time the nearest sample is found, and a window of
    n_pre samples before and n_post samples after that sample is extracted.
    Epochs that would require out-of-bounds indexing are dropped.
    """
    signal   = np.asarray(signal)
    time_vec = np.asarray(time_vec)
    T        = len(signal)

    epochs = []
    for t in anchor_times:
        idx = int(np.argmin(np.abs(time_vec - t)))
        n_pre, n_post = int(t_pre * _HP_FS), int(t_post * _HP_FS)
        i0, i1 = idx - n_pre, idx + n_post
        if i0 >= 0 and i1 <= T:
            epochs.append(signal[i0:i1])

    return np.array(epochs) if epochs else np.empty((0, n_pre + n_post))
