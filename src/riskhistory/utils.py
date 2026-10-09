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