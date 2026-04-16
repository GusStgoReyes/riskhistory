import numpy as np
from sklearn.decomposition import PCA


def _single_prf(max_secs, width, time_to_peak, sampling_rate=25):
    """Compute a single Erlang gamma PRF (same formula as utils_prf.py)."""
    t_ms = np.arange(0, max_secs * 1000, 1000 / sampling_rate)
    signal = (t_ms ** width) * np.exp(-t_ms * width / time_to_peak)
    signal = signal / np.max(signal)
    return signal


def get_prf_basis(max_secs, n_components=3,
                  width_range=(5, 15), time_to_peak_range=(500, 1300),
                  n_grid=20, sampling_rate=25):
    """Generate orthogonal PRF basis functions via PCA over a grid of Erlang gamma PRFs.

    Instead of committing to a single fixed PRF shape, this function builds a
    library of n_grid^2 candidate PRFs by varying the width (shape) and
    time-to-peak parameters across a uniform grid, then extracts the top
    n_components principal components.  The resulting basis spans the space of
    plausible pupil response shapes without assuming any particular one.

    Parameters
    ----------
    max_secs : float
        Duration of each PRF in seconds.
    n_components : int
        Number of PCA components to retain (3 is a reasonable default).
    width_range : tuple of float
        (min, max) range for the Erlang shape parameter (n in Hoeks & Levelt 1993).
    time_to_peak_range : tuple of float
        (min, max) range for the time-to-peak in milliseconds.
    n_grid : int
        Grid resolution along each parameter dimension (total PRFs = n_grid^2).
    sampling_rate : int
        Sampling rate in Hz.

    Returns
    -------
    basis : np.ndarray, shape (T, n_components)
        Orthogonal basis vectors. Column 0 is the dominant (canonical-like)
        component; subsequent columns capture timing and dispersion variability.
    explained_variance_ratio : np.ndarray, shape (n_components,)
        Fraction of variance in the PRF library explained by each component.
    """
    widths = np.linspace(width_range[0], width_range[1], n_grid)
    time_to_peaks = np.linspace(time_to_peak_range[0], time_to_peak_range[1], n_grid)

    prfs = []
    for w in widths:
        for ttp in time_to_peaks:
            prf = _single_prf(max_secs, width=w, time_to_peak=ttp, sampling_rate=sampling_rate)
            prfs.append(prf)

    prf_matrix = np.array(prfs)  # (n_grid^2, T)

    pca = PCA(n_components=n_components)
    pca.fit(prf_matrix)

    basis = pca.components_.T  # (T, n_components)

    # Orient each component so its peak deflection is positive (sign is arbitrary in PCA)
    for i in range(n_components):
        peak_idx = np.argmax(np.abs(basis[:, i]))
        if basis[peak_idx, i] < 0:
            basis[:, i] = -basis[:, i]

    return basis, pca.explained_variance_ratio_
