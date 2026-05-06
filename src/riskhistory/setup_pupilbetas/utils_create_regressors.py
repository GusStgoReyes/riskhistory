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
            timings_matrix[indx_stime:indx_etime, indx_trial] = 1 / (indx_etime - indx_stime)
           
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
                timings_matrix[indx_stime:indx_etime, 0] = 1 / (indx_etime - indx_stime)
            else:
                timings_matrix[indx_stime:indx_etime, 1] = 1 / (indx_etime - indx_stime)

    elif option == "all":
        labels = [f"{regressor_name}-{dataset}_sub-{sub}_run-{run}"]
        timings_matrix = np.zeros((len(time_vector), 1))
        for indx_trial, time in enumerate(timings):
            indx_stime = np.argmin(np.abs(time - time_vector))
            if durations[indx_trial] == 0:
                indx_etime = indx_stime + 1
            else:
                indx_etime = np.argmin(np.abs((time + durations[indx_trial]) - time_vector))
            timings_matrix[indx_stime:indx_etime, 0] = 1 / (indx_etime - indx_stime)
    else:
        raise ValueError("Option must be one of 'ls-s', 'ls-a', or 'all'.")
    
    return timings_matrix, labels

def create_fixation_matrix(eye_x, 
                           eye_y):
    # Z-score eye_x and eye_y
    eye_x = (eye_x - np.mean(eye_x)) / np.std(eye_x)
    eye_y = (eye_y - np.mean(eye_y)) / np.std(eye_y)

    # Remove outliers
    eye_x = np.where(np.abs(eye_x) > 3, 0, eye_x)
    eye_y = np.where(np.abs(eye_y) > 3, 0, eye_y)

    # If either coordinate is 0 (outlier), zero both
    zero_mask = (eye_x == 0) | (eye_y == 0)
    eye_x = np.where(zero_mask, 0, eye_x)
    eye_y = np.where(zero_mask, 0, eye_y)
    
    fixation_matrix = np.zeros((len(eye_x), 4))

    fixation_matrix[:, 0] = eye_x
    fixation_matrix[:, 1] = eye_y
    fixation_matrix[:, 2] = np.square(eye_x)
    fixation_matrix[:, 3] = np.square(eye_y)

    labels = ["eye_x", "eye_y", "eye_x_squared", "eye_y_squared"]
    return fixation_matrix, labels

def convolve_with_prf(timings_matrix, prf_vector):
    convolved_matrix = np.zeros_like(timings_matrix)
    for i in range(timings_matrix.shape[1]):
        convolved_matrix[:, i] = np.convolve(timings_matrix[:, i], prf_vector, mode = "full")[:len(timings_matrix)]
    return convolved_matrix

