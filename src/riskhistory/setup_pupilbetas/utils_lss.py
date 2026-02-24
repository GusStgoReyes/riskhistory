import numpy as np
from riskhistory.setup_pupilbetas.utils_create_regressors import (create_timings_matrix,
                                                                    create_fixation_matrix,
                                                                    convolve_with_prf)

from riskhistory.setup_pupilbetas.utils_models import gls

import pandas as pd
import matplotlib.pyplot as plt

def build_shared_design_matrix_run(all_pupil_data,
                                    chosen_prf,
                                    start_timings, 
                                    response_timings,
                                    sub,
                                    run,
                                    dataset):

    time_vector = all_pupil_data[run]["time_secs"].values
    run_start_timings = start_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    run_response_timings = response_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()

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

    # Fixation regressors (4 betas estimated per run)
    fixation_regressors, fixation_betas_name = create_fixation_matrix(all_pupil_data[run]["eye_x"].values, 
                                                                        all_pupil_data[run]["eye_y"].values)
    
    # Intercept regressor (1 beta estimated per run)
    intercept_regressor = np.ones((len(time_vector), 1))
    intercept_betas_name = [f"intercept-{dataset}_sub-{sub}_run-{run}"]

    # Combine all regressors into a single design matrix
    design_matrix = np.hstack((onset_regressors, response_regressors, fixation_regressors, intercept_regressor))
    betas_name = onset_betas_name + response_betas_name + fixation_betas_name + intercept_betas_name

    return design_matrix, betas_name

def build_trial_regressor(all_pupil_data,
                            chosen_prf,
                            start_timings, 
                            sub,
                            run,
                            dataset,
                            trial_of_interest):
    time_vector = all_pupil_data[run]["time_secs"].values
    run_start_timings = start_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    durations = start_timings.query("run == @run").set_index("trial")["duration"].to_dict()
    # Trial-specific regressors (2 betas estimated per trial: trial of interest vs other trials)
    trial_regressors, trial_betas_name = create_timings_matrix(run_start_timings, 
                                                                time_vector, 
                                                                trial_durations = durations,
                                                                option = "ls-s",
                                                                trial_of_interest = trial_of_interest,
                                                                sub = sub,
                                                                run = run,
                                                                dataset = dataset,
                                                                regressor_name = "decision")
    trial_regressors = convolve_with_prf(trial_regressors, chosen_prf)

    return trial_regressors, trial_betas_name

def lss_single_trial(all_pupil_data,
                    shared_design_matrix,
                    chosen_prf,
                    start_timings, 
                    sub,
                    run,
                    dataset,
                    trial_of_interest):
    design_matrix, trial_betas_names = shared_design_matrix
    trial_regressors, trial_betas_name = build_trial_regressor(all_pupil_data,
                                                                chosen_prf,
                                                                start_timings, 
                                                                sub,
                                                                run,
                                                                dataset,
                                                                trial_of_interest)
    # Combine the shared design matrix with the trial-specific regressors
    full_design_matrix = np.hstack((design_matrix, trial_regressors))
    full_betas_name = trial_betas_names + trial_betas_name

    # Fit the GLM to get the beta estimates for this trial
    y = all_pupil_data[run]["pupil_filtered"].values
    betas_estimates, _ = gls(full_design_matrix, y)
    
    return betas_estimates, full_betas_name

def lss_single_run(all_pupil_data,
                   chosen_prf,
                   start_timings, 
                   response_timings,
                   sub,
                   run,
                   dataset):
    start_timings_run = start_timings.query("run == @run")
    response_timings_run = response_timings.query("run == @run")
        
    shared_design_matrix = build_shared_design_matrix_run(all_pupil_data,
                                                            chosen_prf,
                                                            start_timings_run, 
                                                            response_timings_run,
                                                            sub,
                                                            run,
                                                            dataset)
    results_run = {"target_trial" : [],
                   "trial_betas_estimates" : [],
                   "trial_betas_names" : []}
    
    for trial in start_timings_run["trial"].unique():
        betas_estimates, full_betas_name = lss_single_trial(all_pupil_data,
                                                            shared_design_matrix,
                                                            chosen_prf,
                                                            start_timings_run, 
                                                            sub,
                                                            run,
                                                            dataset,
                                                            trial_of_interest = trial)
        results_run["target_trial"].extend([trial] * len(full_betas_name))  
        results_run["trial_betas_estimates"].extend(betas_estimates.tolist())  # Convert to list for easier storage in dataframe
        results_run["trial_betas_names"].extend(full_betas_name)
        
    return pd.DataFrame(results_run)

def lss_single_sub(all_pupil_data,
                    chosen_prf,
                    start_timings, 
                    response_timings,
                    sub,
                    runs,
                    dataset):
    
    results_sub = pd.DataFrame()
    for run in runs:
        results_run = lss_single_run(all_pupil_data,
                                    chosen_prf,
                                    start_timings, 
                                    response_timings,
                                    sub,
                                    run,
                                    dataset)
        results_run["run"] = run
        results_sub = pd.concat([results_sub, results_run], ignore_index=True)
    
    results_sub["sub"] = sub
    results_sub["dataset"] = dataset
    
    return results_sub