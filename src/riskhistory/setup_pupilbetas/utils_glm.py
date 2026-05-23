import numpy as np
from scipy.linalg import block_diag
from riskhistory.setup_pupilbetas.utils_create_regressors import (create_timings_matrix,
                                                                    create_fixation_matrix,
                                                                    convolve_with_prf)
from riskhistory.setup_pupilbetas.utils_get_data import get_response_timings_df
from riskhistory.setup_pupilbetas.utils_models import ridge
import pandas as pd


def compute_vif(design_matrix, betas_name, sub, run, dataset, regressor_type = "decision"):
    """
        Compute VIF for regressors of interest within the full design matrix.
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
    vif = full_vif[regressor_mask]
    
    return {**base,
            f"mean_vif_{regressor_type}": float(np.nanmean(vif)),
            f"sd_vif_{regressor_type}":   float(np.nanstd(vif))}
    
def build_design_matrix_run(all_pupil_data,
                            chosen_prf,
                            start_timings,
                            sub,
                            run,
                            dataset):

    time_vector = all_pupil_data[run]["time_secs"].values
    run_start_timings = start_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    durations = start_timings.query("run == @run").set_index("trial")["duration"].to_dict()

    # Onset regressors (1 beta estimated per run for onset)
    onset_regressors, onset_betas_name = create_timings_matrix(run_start_timings,
                                                                time_vector,
                                                                trial_durations=None,
                                                                option="all",
                                                                trial_of_interest=None,
                                                                sub=sub,
                                                                run=run,
                                                                dataset=dataset,
                                                                regressor_name="onset")
    onset_regressors = convolve_with_prf(onset_regressors, chosen_prf)

    # Decision regressors (1 beta per trial, spanning stimulus onset to response)
    decision_regressors, decision_betas_name = create_timings_matrix(run_start_timings,
                                                                      time_vector,
                                                                      trial_durations=durations,
                                                                      option="ls-a",
                                                                      trial_of_interest=None,
                                                                      sub=sub,
                                                                      run=run,
                                                                      dataset=dataset,
                                                                      regressor_name="decision")
    decision_regressors = convolve_with_prf(decision_regressors, chosen_prf)

    # Button press regressors (1 beta per run)
    response_timings_run = (get_response_timings_df(dataset)
                            .query("sub == @sub and run == @run")
                            .set_index("trial")["time_secs"]
                            .to_dict())
    button_regressors, button_betas_name = create_timings_matrix(response_timings_run,
                                                                  time_vector,
                                                                  trial_durations=None,
                                                                  option="all",
                                                                  trial_of_interest=None,
                                                                  sub=sub,
                                                                  run=run,
                                                                  dataset=dataset,
                                                                  regressor_name="button")
    button_regressors = convolve_with_prf(button_regressors, chosen_prf)

    # Fixation regressors (4 betas estimated per run)
    fixation_regressors, fixation_betas_name = create_fixation_matrix(all_pupil_data[run]["eye_x"].values,
                                                                       all_pupil_data[run]["eye_y"].values)

    # Intercept regressor (1 beta estimated per run)
    intercept_regressor = np.ones((len(time_vector), 1))
    intercept_betas_name = [f"intercept-{dataset}_sub-{sub}_run-{run}"]

    # Linear trend regressor
    linear_trend = np.arange(len(time_vector)).reshape(-1, 1)
    # zscore the linear trend regressor
    linear_trend = (linear_trend - linear_trend.mean()) / linear_trend.std()
    linear_trend_name = [f"linear_trend-{dataset}_sub-{sub}_run-{run}"]

    design_matrix = np.hstack((onset_regressors, 
                               fixation_regressors, 
                               intercept_regressor, 
                               decision_regressors, 
                               button_regressors, 
                               linear_trend))
    
    betas_name = (onset_betas_name + 
                  fixation_betas_name + 
                  intercept_betas_name + 
                  decision_betas_name + 
                  button_betas_name + 
                  linear_trend_name)


    return design_matrix, betas_name

def glm_single_run(all_pupil_data,
                   chosen_prf,
                   start_timings,
                   sub,
                   run,
                   dataset):
    start_timings_run = start_timings.query("run == @run")

    design_matrix, betas_name = build_design_matrix_run(all_pupil_data,
                                                        chosen_prf,
                                                        start_timings_run,
                                                        sub,
                                                        run,
                                                        dataset)
    y = all_pupil_data[run]["pupil"].values
    betas_estimates, residuals, best_alpha = ridge(design_matrix, y)

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"R-squared for sub-{sub} run-{run}: {r2:.4f}")

    n = len(betas_name)
    results_run = {
        "betas_estimates": betas_estimates.tolist(),
        "betas_names": betas_name,
        "r2": [r2] * n,
        "alpha": [best_alpha] * n,
    }

    vif_stats = compute_vif(design_matrix, betas_name, sub, run, dataset, regressor_type="decision")
    print(f"  Decision VIF — mean: {vif_stats['mean_vif_decision']:.2f}, "
          f"sd: {vif_stats['sd_vif_decision']:.2f} (n={vif_stats['n_regressors_decision']})")

    return pd.DataFrame(results_run), vif_stats


def build_design_matrix_all_runs(all_pupil_data,
                                 chosen_prf,
                                 start_timings,
                                 sub,
                                 runs,
                                 dataset):
    dms = []
    all_names = []
    for run in runs:
        dm_run, names_run = build_design_matrix_run(
            all_pupil_data, chosen_prf, start_timings, sub, run, dataset
        )
        dms.append(dm_run)
        all_names.extend(names_run)
    return block_diag(*dms), all_names


def glm_all_runs(all_pupil_data,
                 chosen_prf,
                 start_timings,
                 sub,
                 runs,
                 dataset):
    combined_dm, betas_name = build_design_matrix_all_runs(
        all_pupil_data, chosen_prf, start_timings, sub, runs, dataset
    )
    y = np.concatenate([all_pupil_data[run]["pupil"].values for run in runs])

    betas_estimates, residuals, best_alpha = ridge(combined_dm, y)

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"R-squared for sub-{sub} (all runs combined): {r2:.4f}")

    n = len(betas_name)
    results_df = pd.DataFrame({
        "betas_estimates": betas_estimates.tolist(),
        "betas_names": betas_name,
        "r2": [r2] * n,
        "alpha": [best_alpha] * n,
    })
    results_df["run"] = results_df["betas_names"].str.extract(r'_run-(\d+)')[0].astype("Int64")
    results_df["sub"] = sub
    results_df["dataset"] = dataset

    vif_stats = compute_vif(combined_dm, betas_name, sub, run=None, dataset=dataset,
                            regressor_type="decision")
    print(f"  Decision VIF — mean: {vif_stats['mean_vif_decision']:.2f}, "
          f"sd: {vif_stats['sd_vif_decision']:.2f} (n={vif_stats['n_regressors_decision']})")

    return results_df, vif_stats


def glm_single_sub(all_pupil_data,
                   chosen_prf,
                   start_timings,
                   sub,
                   runs,
                   dataset):
    results_sub, vif_stats = glm_all_runs(
        all_pupil_data, chosen_prf, start_timings, sub, runs, dataset
    )
    return results_sub, pd.DataFrame([vif_stats])