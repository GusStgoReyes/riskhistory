import numpy as np
from riskhistory.setup_pupilbetas.utils_create_regressors import (create_timings_matrix,
                                                                    create_fixation_matrix,
                                                                    convolve_with_prf)

from riskhistory.setup_pupilbetas.utils_models import gls

import pandas as pd
import matplotlib.pyplot as plt

def build_design_matrix_run(all_pupil_data,
                            chosen_prf,
                            start_timings, 
                            response_timings,
                            sub,
                            run,
                            dataset):

    time_vector = all_pupil_data[run]["time_secs"].values
    run_start_timings = start_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    run_response_timings = response_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    durations = start_timings.query("run == @run").set_index("trial")["duration"].to_dict()

    # Onset regressors (1 beta estimated per run for onset)
    onset_regressors, onset_betas_name = create_timings_matrix(run_start_timings, 
                                                                time_vector, 
                                                                trial_durations = None, 
                                                                option = "all", 
                                                                trial_of_interest = None, 
                                                                sub = sub, 
                                                                run = run, 
                                                                dataset = dataset, 
                                                                regressor_name = "onset")
    onset_regressors = convolve_with_prf(onset_regressors, chosen_prf)
    
    # Response regressors (1 beta estimated per run for response)
    response_regressors, response_betas_name = create_timings_matrix(run_response_timings, 
                                                                    time_vector, 
                                                                    trial_durations = None, 
                                                                    option = "all", 
                                                                    trial_of_interest = None, 
                                                                    sub = sub, 
                                                                    run = run, 
                                                                    dataset = dataset, 
                                                                    regressor_name = "response")
    response_regressors = convolve_with_prf(response_regressors, chosen_prf)

    # Decision regressors (1 beta estimated per run for decision)
    decision_regressors, decision_betas_name = create_timings_matrix(run_start_timings, 
                                                                    time_vector, 
                                                                    trial_durations = durations,
                                                                    option = "all", 
                                                                    trial_of_interest = None,
                                                                    sub = sub,
                                                                    run = run,
                                                                    dataset = dataset,
                                                                    regressor_name = "decision")

    # Fixation regressors (4 betas estimated per run)
    fixation_regressors, fixation_betas_name = create_fixation_matrix(all_pupil_data[run]["eye_x"].values, 
                                                                        all_pupil_data[run]["eye_y"].values)
    
    # Intercept regressor (1 beta estimated per run)
    intercept_regressor = np.ones((len(time_vector), 1))
    intercept_betas_name = [f"intercept-{dataset}_sub-{sub}_run-{run}"]

    # Combine all regressors into a single design matrix
    design_matrix = np.hstack((onset_regressors, response_regressors, fixation_regressors, intercept_regressor, decision_regressors))
    betas_name = onset_betas_name + response_betas_name + fixation_betas_name + intercept_betas_name + decision_betas_name

    return design_matrix, betas_name

def glm_single_run(all_pupil_data,
                   chosen_prf,
                   start_timings, 
                   response_timings,
                   sub,
                   run,
                   dataset):
    start_timings_run = start_timings.query("run == @run")
    response_timings_run = response_timings.query("run == @run")
        
    design_matrix, betas_name = build_design_matrix_run(all_pupil_data,
                                                        chosen_prf,
                                                        start_timings_run, 
                                                        response_timings_run,
                                                        sub,
                                                        run,
                                                        dataset)
    # Fit the GLM to get the beta estimates for this trial
    y = all_pupil_data[run]["pupil_filtered"].values
    betas_estimates, residuals = gls(design_matrix, y)

    results_run = {
        "betas_estimates": betas_estimates.tolist(),
        "betas_names": betas_name,
    }
        
    return pd.DataFrame(results_run)