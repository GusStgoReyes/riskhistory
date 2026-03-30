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

def ols(X, y):
    """Perform Ordinary Least Squares regression."""
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_ols
    return beta_ols, residuals

def ridge(X, y):
    # Cross-validated Ridge to find best alpha
    alphas = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X, y)
    
    # Get coefficients
    beta_estimates = ridge_cv.coef_
    
    # Compute residuals
    residuals = y - ridge_cv.predict(X)
    
    return beta_estimates, residuals

def gls(X, y):
    """Perform Generalized Least Squares regression to account for autocorrelation in the residuals."""
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_ols
    rho = np.corrcoef(residuals[:-1], residuals[1:])[0,1]
    y_whitened = y[1:] - rho * y[:-1]
    X_whitened = X[1:] - rho * X[:-1]
    beta_gls = np.linalg.lstsq(X_whitened, y_whitened, rcond=None)[0]

    residuals = y - X @ beta_gls

    return beta_gls, residuals

def durbin_watson(residuals):
    """Quick check for autocorrelation"""
    diff_resid = np.diff(residuals)
    dw = np.sum(diff_resid**2) / np.sum(residuals**2)
    return dw  # ~2 means no autocorrelation, <2 positive, >2 negative