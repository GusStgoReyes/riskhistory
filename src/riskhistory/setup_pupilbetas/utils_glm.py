import numpy as np
from riskhistory.setup_pupilbetas.utils_create_regressors import (create_timings_matrix,
                                                                    create_fixation_matrix,
                                                                    convolve_with_prf)

from riskhistory.setup_pupilbetas.utils_models import ridge
import pandas as pd


def compute_vif(design_matrix, betas_name, sub, run, dataset, regressor_type = "decision"):
    """Compute VIF for regressors of interest within the full design matrix.

    Uses the correlation-matrix formula: VIF_j = (R^{-1})_{jj}, where R is
    the correlation matrix of all design-matrix columns. This accounts for
    collinearity of each decision regressor against both the other decision
    regressors AND the nuisance regressors (onset, response, fixation,
    intercept).

    Parameters
    ----------
    design_matrix : np.ndarray, shape (T, P)
    betas_name    : list of str, length P
    sub, run, dataset : identifiers written into the output dict

    Returns
    -------
    dict with keys: sub, run, dataset, mean_vif, sd_vif, n_regressors
    """
    regressor_mask = np.array([regressor_type in name for name in betas_name])
    n_regressors = int(regressor_mask.sum())
    base = {"sub": sub, "run": run, "dataset": dataset, f"n_regressors_{regressor_type}": n_regressors}

    if n_regressors < 2:
        return {**base, f"mean_vif_{regressor_type}": float("nan"), f"sd_vif_{regressor_type}": float("nan")}

    X = design_matrix.astype(float)
    X = X - X.mean(axis=0)
    col_std = X.std(axis=0, ddof=1)
    non_constant_mask = col_std > 1e-12
    X_std = X[:, non_constant_mask] / col_std[non_constant_mask]
    R = (X_std.T @ X_std) / (X_std.shape[0] - 1)
    R_inv = np.linalg.inv(R) #np.linalg.pinv(R)

    # map back
    full_vif = np.full(len(col_std), np.nan)
    full_vif[non_constant_mask] = np.diag(R_inv)
    vif_decision = full_vif[regressor_mask]
    
    return {**base,
            f"mean_vif_{regressor_type}": float(np.mean(vif_decision)),
            f"sd_vif_{regressor_type}":   float(np.std(vif_decision))}
    
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
    
    # Response regressors (num trials beta estimated per run for response)
    response_regressors, response_betas_name = create_timings_matrix(run_response_timings, 
                                                                    time_vector, 
                                                                    trial_durations = None, 
                                                                    option = "ls-a", 
                                                                    trial_of_interest = None, 
                                                                    sub = sub, 
                                                                    run = run, 
                                                                    dataset = dataset, 
                                                                    regressor_name = "response")
    response_regressors = convolve_with_prf(response_regressors, chosen_prf)

    # Decision regressors (num trials beta estimated per run for decision)
    decision_regressors, decision_betas_name = create_timings_matrix(run_start_timings,
                                                                    time_vector,
                                                                    trial_durations = durations,
                                                                    option = "ls-a",
                                                                    trial_of_interest = None,
                                                                    sub = sub,
                                                                    run = run,
                                                                    dataset = dataset,
                                                                    regressor_name = "decision")
    decision_regressors = convolve_with_prf(decision_regressors, chosen_prf)

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
    y = all_pupil_data[run]["pupil"].values
    betas_estimates, residuals = ridge(design_matrix, y)

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"R-squared for sub-{sub} run-{run}: {r2:.4f}")

    results_run = {
        "betas_estimates": betas_estimates.tolist(),
        "betas_names": betas_name,
        "r2": [r2] * len(betas_name),
    }

    vif_stats_decision = compute_vif(design_matrix, betas_name, sub, run, dataset, regressor_type="decision")
    print(f"  Decision VIF — mean: {vif_stats_decision['mean_vif_decision']:.2f}, "
          f"sd: {vif_stats_decision['sd_vif_decision']:.2f} (n={vif_stats_decision['n_regressors_decision']})")
    
    vif_stats_response = compute_vif(design_matrix, betas_name, sub, run, dataset, regressor_type="response")
    print(f"  Response VIF — mean: {vif_stats_response['mean_vif_response']:.2f}, "
          f"sd: {vif_stats_response['sd_vif_response']:.2f} (n={vif_stats_response['n_regressors_response']})")
    vif_stats = {**vif_stats_decision, **vif_stats_response}
    return pd.DataFrame(results_run), vif_stats


def glm_single_sub(all_pupil_data,
                   chosen_prf,
                   start_timings,
                   response_timings,
                   sub,
                   runs,
                   dataset):
    results_sub = pd.DataFrame()
    vif_records = []
    for run in runs:
        results_run, vif_stats = glm_single_run(all_pupil_data, chosen_prf,
                                                start_timings, response_timings,
                                                sub, run, dataset)
        results_run["run"] = run
        results_sub = pd.concat([results_sub, results_run], ignore_index=True)
        vif_records.append(vif_stats)
    results_sub["sub"] = sub
    results_sub["dataset"] = dataset
    return results_sub, pd.DataFrame(vif_records)