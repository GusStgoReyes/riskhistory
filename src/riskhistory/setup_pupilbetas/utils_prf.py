import numpy as np

def get_prf(max_secs, width = 10.1, time_to_peak = 930, sampling_rate = 25):
    """Compute the pupil response function (PRF) at time t with parameters width and time_to_peak.
     """
    t_ms = np.arange(0, max_secs * 1000, 1000/sampling_rate)  # Time vector in milliseconds
    signal = (t_ms ** width) * np.exp(-t_ms * width / time_to_peak)
    signal = signal / np.max(signal)  # Normalize the PRF to have a maximum
    return signal