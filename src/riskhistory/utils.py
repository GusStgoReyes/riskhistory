import pandas as pd
import numpy as np
import itertools
from scipy.stats import spearmanr

def get_trials(dataset_name):
    """
    Get all gain-loss combinations for a given dataset.
    """
    if dataset_name == "equalIndifference":
        gains = np.arange(10, 41, 2)
        losses = np.arange(5, 21, 1)
        combinations = list(itertools.product(gains, losses))
    elif dataset_name == "equalRange":
        gains = np.arange(5, 21, 1)
        losses = np.arange(5, 21, 1)
        combinations = list(itertools.product(gains, losses))
    elif dataset_name == "sheng2020":
        gains = np.arange(1, 11, 1)
        losses = np.arange(1, 11, 1)
        combinations = list(itertools.product(gains, losses))
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    
    df = pd.DataFrame(combinations, columns=['gain', 'loss'])
    return df

def bootstrap_spearman_difference(
    x,
    y1,
    y2,
    n_boot=10000,
    alpha=0.05,
    random_state=None
):
    """
    Basic percentile bootstrap CI for the difference between
    two overlapping Spearman correlations.

    Tests the difference:
        rho(X, Y1) - rho(X, Y2)
    """

    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    n = len(x)

    if n < 3:
        raise ValueError("Need at least 3 complete observations.")

    # Observed correlations
    r1 = spearmanr(x, y1).statistic
    r2 = spearmanr(x, y2).statistic
    observed_diff = r1 - r2

    # Bootstrap
    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.empty(n_boot)

    for b in range(n_boot):

        # Resample PARTICIPANTS jointly
        indices = rng.integers(0, n, size=n)

        xb = x[indices]
        y1b = y1[indices]
        y2b = y2[indices]

        r1b = spearmanr(xb, y1b).statistic
        r2b = spearmanr(xb, y2b).statistic

        bootstrap_diffs[b] = r1b - r2b

    # Percentile bootstrap CI
    lower = np.quantile(
        bootstrap_diffs,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_diffs,
        1 - alpha / 2
    )

    return {
        "r_x_y1": r1,
        "r_x_y2": r2,
        "difference": observed_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_differences": bootstrap_diffs,
        "n": n,
    }

def flag_outliers_mad(values, k=3):
    """|x - median| > k scaled MADs (1.4826 * MAD, the SD under normality)."""
    deviation = np.abs(values - values.median())
    scaled_mad = 1.4826 * deviation.median()
    if scaled_mad == 0:
        return pd.Series(False, index=values.index)
    return deviation / scaled_mad > k


def remove_outliers_mad(df, columns, k=3, group_col="dataset", id_col="sub"):
    """Drop rows that are MAD outliers on any of `columns`, within each `group_col`.
    Returns the outlier-free dataframe and a per-group summary (n_total,
    n_outliers, outlier ids)."""
    is_outlier = pd.concat(
        [df.groupby(group_col)[c].transform(flag_outliers_mad, k=k) for c in columns], axis=1
    ).any(axis=1)

    summary = (df.assign(is_outlier=is_outlier)
               .groupby(group_col, sort=False)
               .apply(lambda g: pd.Series({
                   "n_total": len(g),
                   "n_outliers": int(g["is_outlier"].sum()),
                   "outliers": g.loc[g["is_outlier"], id_col].tolist(),
               }), include_groups=False)
               .reset_index())
    return df[~is_outlier], summary
