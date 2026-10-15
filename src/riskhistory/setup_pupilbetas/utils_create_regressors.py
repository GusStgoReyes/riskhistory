import numpy as np

def create_timings_matrix(trial_timings,
                          time_vector,
                          trial_durations = None,
                          option = "ls-a",
                          trial_of_interest = None,
                          sub = None,
                          run = None,
                          dataset = None,
                          regressor_name = None):
    timings = [trial_timings[t] for t in sorted(trial_timings.keys())]
    trials = sorted(trial_timings.keys())
    if trial_durations is not None:
        durations = [trial_durations[t] if t in trial_durations else 0 for t in sorted(trial_timings.keys())]
    else:
        durations = [0 for t in sorted(trial_timings.keys())]

    # Initialize the design matrix
    if option == "ls-a":
        labels = [f"{regressor_name}-{dataset}_sub-{sub}_run-{run}_trial-{trial}" for trial in trials]
        timings_matrix = np.zeros((len(time_vector), len(timings)))
        for indx_trial, time in enumerate(timings):
            indx_stime = np.argmin(np.abs(time - time_vector))
            if durations[indx_trial] == 0:
                indx_etime = indx_stime + 1
            else:
                indx_etime = np.argmin(np.abs((time + durations[indx_trial]) - time_vector))
            timings_matrix[indx_stime:indx_etime, indx_trial] = 1
           
    elif option == "ls-s":
        labels = [f"target_{regressor_name}-{dataset}_sub-{sub}_run-{run}_trial-{trial_of_interest}", f"{regressor_name}-{dataset}_sub-{sub}_run-{run}_trial-other"]
        timings_matrix = np.zeros((len(time_vector), 2))
        for indx_trial, time in enumerate(timings):
            indx_stime = np.argmin(np.abs(time - time_vector))
            if durations[indx_trial] == 0:
                indx_etime = indx_stime + 1
            else:
                indx_etime = np.argmin(np.abs((time + durations[indx_trial]) - time_vector))

            if trials[indx_trial] == trial_of_interest:
                timings_matrix[indx_stime:indx_etime, 0] = 1
            else:
                timings_matrix[indx_stime:indx_etime, 1] = 1

    elif option == "all":
        labels = [f"{regressor_name}-{dataset}_sub-{sub}_run-{run}"]
        timings_matrix = np.zeros((len(time_vector), 1))
        for indx_trial, time in enumerate(timings):
            indx_stime = np.argmin(np.abs(time - time_vector))
            if durations[indx_trial] == 0:
                indx_etime = indx_stime + 1
            else:
                indx_etime = np.argmin(np.abs((time + durations[indx_trial]) - time_vector))
            timings_matrix[indx_stime:indx_etime, 0] = 1
    else:
        raise ValueError("Option must be one of 'ls-s', 'ls-a', or 'all'.")
    
    return timings_matrix, labels

def create_fixation_matrix(eye_x,
                           eye_y,
                           run_block,
                           outlier_z = 3):
    eye_x = np.asarray(eye_x, dtype=float)
    eye_y = np.asarray(eye_y, dtype=float)

    # Missing gaze is stored as raw 0 by setup_pupildata (NaN -> 0); a sample
    # is invalid if either coordinate is missing
    invalid = (eye_x == 0) | (eye_y == 0) | np.isnan(eye_x) | np.isnan(eye_y)

    labels = [f"eye_x_run-{run_block}", f"eye_y_run-{run_block}", f"eye_x_squared_run-{run_block}", f"eye_y_squared_run-{run_block}"]

    # No usable gaze: all-zero columns (no gaze information) rather than NaNs,
    # which the ridge fit cannot handle
    if (~invalid).sum() < 2:
        return np.zeros((len(eye_x), 4)), labels

    # Z-score using valid samples only, so missing samples don't inflate the SD
    def zscore_valid(v, valid):
        return (v - np.mean(v[valid])) / np.std(v[valid])

    eye_x = zscore_valid(eye_x, ~invalid)
    eye_y = zscore_valid(eye_y, ~invalid)

    # Outliers on either coordinate invalidate the sample
    invalid |= (np.abs(eye_x) > outlier_z) | (np.abs(eye_y) > outlier_z)

    # Squares from the z-scores; mean-impute invalid samples so they carry no
    # gaze information: 0 for the linear terms, mean of valid squares for the
    # quadratic terms (0 would code them as perfectly centered gaze)
    eye_x_sq = np.square(eye_x)
    eye_y_sq = np.square(eye_y)
    eye_x_sq[invalid] = np.mean(eye_x_sq[~invalid])
    eye_y_sq[invalid] = np.mean(eye_y_sq[~invalid])
    eye_x[invalid] = 0
    eye_y[invalid] = 0

    fixation_matrix = np.zeros((len(eye_x), 4))

    fixation_matrix[:, 0] = eye_x
    fixation_matrix[:, 1] = eye_y
    fixation_matrix[:, 2] = eye_x_sq
    fixation_matrix[:, 3] = eye_y_sq

    return fixation_matrix, labels

def convolve_with_prf(timings_matrix, prf_vector, normalize):
    convolved_matrix = np.zeros_like(timings_matrix)
    for i in range(timings_matrix.shape[1]):
        convolved_matrix[:, i] = np.convolve(timings_matrix[:, i], prf_vector, mode = "full")[:len(timings_matrix)]
        if normalize:
            # Make max be 1
            convolved_matrix[:, i] = convolved_matrix[:, i] / np.max(convolved_matrix[:, i])
    return convolved_matrix

