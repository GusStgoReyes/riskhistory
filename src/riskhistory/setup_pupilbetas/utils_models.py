import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import RidgeCV

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
    

def ridge(X, y, alphas=None, n_splits=3):
    if alphas is None:
        alphas = np.logspace(-5, 0, 25)

    alphas = np.logspace(-3, 6, 50)
    model = RidgeCV(alphas=alphas, cv=None, fit_intercept = False)
    model.fit(X, y)

    print(f"Best alpha: {model.alpha_}")
    y_pred = model.predict(X)
    residuals = y - y_pred

    return model.coef_, residuals, model.alpha_