import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

def highpass_filter(data, cutoff=0.01, fs=25, order=3):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq

    # Design Butterworth high-pass filter
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply filter (filtfilt does forward-backward filtering for zero phase shift)
    filtered = filtfilt(b, a, data)

    return filtered

def lowpass_filter(data, cutoff=0.05, fs=25, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered

def ols(X, y):
    """Perform Ordinary Least Squares regression."""
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_ols
    return beta_ols, residuals
    

def ridge(X, y, alphas=None, n_splits=5):
    if alphas is None:
        alphas = np.logspace(-5, -1, 25)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mean_errors = []
    for alpha in alphas:
        fold_errors = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_errors.append(np.mean((y_test - y_pred) ** 2))
        mean_errors.append(np.mean(fold_errors))

    best_alpha = alphas[np.argmin(mean_errors)]
    print(f"Best alpha: {best_alpha:.5f} with MSE: {min(mean_errors):.5f}")

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X, y)
    y_pred = final_model.predict(X)
    residuals = y - y_pred

    return final_model.coef_, residuals, best_alpha