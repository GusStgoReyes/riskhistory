"""Exponentially decayed choice history.

choice_history_t = sum_{k=1}^{min(max_lag, t-1)} (1-eta) * eta^(k-1) * choice_{t-k}

i.e. a decayed sum of the past `max_lag` choices at most (fewer if fewer
trials have happened yet in the run). Trials beyond max_lag back do not
contribute at all, so the weights sum to (1 - eta^max_lag) rather than 1
once a full window is available.

Should be called on the full (pre-exclusion) data: a trial later excluded
for RT reasons still really happened, so it should still contribute to the
history available to subsequent trials in the run. Apply any `exclude`
filtering after this, not before.
"""

import numpy as np
import pandas as pd


def compute_choice_history(df, decay_rate, max_lag=7):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns dataset, sub, run, trial_run, choice.
    decay_rate : float or dict
        A single rate applied to all datasets, or a dict mapping
        dataset -> rate.
    max_lag : int
        Maximum number of previous trials that can contribute to the
        history (default 7).

    Returns
    -------
    pd.DataFrame
        Copy of df with a new `choice_history` column: the decayed history
        of choices strictly before each trial (NaN for the first trial in
        each run, i.e. no prior history).
    """
    per_dataset = isinstance(decay_rate, dict)
    if per_dataset:
        missing = set(df["dataset"].unique()) - set(decay_rate.keys())
        if missing:
            raise ValueError(f"decay_rate missing for datasets: {missing}")

    df = df.sort_values(["dataset", "sub", "run", "trial_run"]).copy()
    df["_rate"] = df["dataset"].map(decay_rate) if per_dataset else decay_rate

    def _history_for_group(group):
        rate = group["_rate"].iloc[0]
        # A missing/no-response trial's own choice contributes nothing to the
        # history of later trials (consistent with how the old fixed-lag
        # choice_prevN columns were NA-filled to 0).
        choice_filled = group["choice"].fillna(0)
        weights = (1 - rate) * rate ** np.arange(max_lag)

        history = np.zeros(len(choice_filled))
        for k in range(1, max_lag + 1):
            lag_k = choice_filled.shift(k)
            # lag_k is NaN only when trial t-k doesn't exist yet (i.e. fewer
            # than k trials have happened in this run) -- that trial simply
            # doesn't contribute, rather than NaN-ing out the whole sum.
            history += np.where(lag_k.isna(), 0, weights[k - 1] * lag_k)

        history = pd.Series(history, index=group.index)
        # The first trial in the run has no prior trials at all, so its
        # history is undefined (not just a sum of zero terms).
        history.iloc[0] = np.nan
        return history

    df["choice_history"] = (
        df.groupby(["dataset", "sub", "run"], group_keys=False)[["choice", "_rate"]]
        .apply(_history_for_group)
    )
    df = df.drop(columns="_rate")

    return df
