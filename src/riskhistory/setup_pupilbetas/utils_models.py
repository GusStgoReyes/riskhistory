import numpy as np

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