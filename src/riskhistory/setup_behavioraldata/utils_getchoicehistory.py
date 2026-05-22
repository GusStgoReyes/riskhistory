import numpy as np

def compute_choice_history(df, num_prev_trials=3, decay_rate=None):
    """
    Compute an exponentially decayed sum of previous choices.

    choice_history = sum_{k=1}^{num_prev_trials} exp(-decay_rate * k) * choice_prev{k}
    """
    if decay_rate is None:
        raise ValueError("decay_rate must be provided as a dict mapping dataset -> rate")

    missing = set(df['dataset'].unique()) - set(decay_rate.keys())
    if missing:
        raise ValueError(f"decay_rate missing for datasets: {missing}")

    prev_cols = [f'choice_prev{k}' for k in range(1, num_prev_trials + 1)]
    missing_cols = [c for c in prev_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    df['choice_history'] = 0.0
    for dataset, rate in decay_rate.items():
        mask = df['dataset'] == dataset
        # EWMA weights: (1-rate) * rate^(k-1) for k=1,...,N; sum to 1 - rate^N ≈ 1
        weights = np.array([(1 - rate) * np.power(rate, k - 1) for k in range(1, num_prev_trials + 1)])
        df.loc[mask, 'choice_history'] = df.loc[mask, prev_cols].values @ weights

    return df
