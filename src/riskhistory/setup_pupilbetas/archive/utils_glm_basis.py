import numpy as np
import pandas as pd
from riskhistory.setup_pupilbetas.utils_create_regressors import (
    create_timings_matrix,
    create_fixation_matrix,
)
from riskhistory.setup_pupilbetas.utils_models import ridge


def convolve_with_prf_basis(timings_matrix, basis):
    """Convolve each column of timings_matrix with each basis function.

    Parameters
    ----------
    timings_matrix : np.ndarray, shape (T, N)
    basis : np.ndarray, shape (T_basis, K)

    Returns
    -------
    convolved : np.ndarray, shape (T, N * K)
        Columns are interleaved as:
        [col-0-comp-0, col-0-comp-1, ..., col-0-comp-K-1,
         col-1-comp-0, ..., col-N-1-comp-K-1]
    """
    T, N = timings_matrix.shape
    K = basis.shape[1]
    convolved = np.zeros((T, N * K))
    for i in range(N):
        for k in range(K):
            convolved[:, i * K + k] = np.convolve(
                timings_matrix[:, i], basis[:, k], mode="full"
            )[:T]
    return convolved


def expand_labels_for_basis(labels, n_components):
    """Append _component-{k} to each label for k in 0..n_components-1."""
    return [
        f"{label}_component-{k}"
        for label in labels
        for k in range(n_components)
    ]


def compute_vif(design_matrix, betas_name, sub, run, dataset, regressor_type="decision"):
    """Compute VIF for regressors of interest within the full design matrix.

    Identical to utils_glm.compute_vif but uses pinv for robustness against the
    larger, more collinear design matrices produced by the basis-set approach.
    """
    regressor_mask = np.array([regressor_type in name for name in betas_name])
    n_regressors = int(regressor_mask.sum())
    base = {"sub": sub, "run": run, "dataset": dataset,
            f"n_regressors_{regressor_type}": n_regressors}

    if n_regressors < 2:
        return {**base,
                f"mean_vif_{regressor_type}": float("nan"),
                f"sd_vif_{regressor_type}": float("nan")}

    X = design_matrix.astype(float)
    X = X - X.mean(axis=0)
    col_std = X.std(axis=0, ddof=1)
    non_constant_mask = col_std > 1e-12
    X_std = X[:, non_constant_mask] / col_std[non_constant_mask]
    R = (X_std.T @ X_std) / (X_std.shape[0] - 1)
    R_inv = np.linalg.pinv(R)

    full_vif = np.full(len(col_std), np.nan)
    full_vif[non_constant_mask] = np.diag(R_inv)
    vif_decision = full_vif[regressor_mask]

    return {
        **base,
        f"mean_vif_{regressor_type}": float(np.nanmean(vif_decision)),
        f"sd_vif_{regressor_type}": float(np.nanstd(vif_decision)),
    }


def build_design_matrix_run_basis(all_pupil_data, basis,
                                   start_timings, response_timings,
                                   sub, run, dataset):
    """Build a GLM design matrix using a multi-component PRF basis set.

    Onset, response, and decision regressors are each convolved with all
    n_components basis functions.  Fixation and intercept regressors are
    unchanged.
    """
    n_components = basis.shape[1]
    time_vector = all_pupil_data[run]["time_secs"].values
    run_start_timings = start_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    run_response_timings = response_timings.query("run == @run").set_index("trial")["time_secs"].to_dict()
    durations = start_timings.query("run == @run").set_index("trial")["duration"].to_dict()

    # Onset (1 regressor per run → n_components betas per run)
    onset_raw, onset_labels = create_timings_matrix(
        run_start_timings, time_vector, trial_durations=None,
        option="all", sub=sub, run=run, dataset=dataset, regressor_name="onset",
    )
    onset_regressors = convolve_with_prf_basis(onset_raw, basis)
    onset_betas_name = expand_labels_for_basis(onset_labels, n_components)

    # Response (1 regressor per trial → n_components betas per trial)
    response_raw, response_labels = create_timings_matrix(
        run_response_timings, time_vector, trial_durations=None,
        option="ls-a", sub=sub, run=run, dataset=dataset, regressor_name="response",
    )
    response_regressors = convolve_with_prf_basis(response_raw, basis)
    response_betas_name = expand_labels_for_basis(response_labels, n_components)

    # Decision (1 regressor per trial → n_components betas per trial)
    decision_raw, decision_labels = create_timings_matrix(
        run_start_timings, time_vector, trial_durations=durations,
        option="ls-a", sub=sub, run=run, dataset=dataset, regressor_name="decision",
    )
    decision_regressors = convolve_with_prf_basis(decision_raw, basis)
    decision_betas_name = expand_labels_for_basis(decision_labels, n_components)

    # Fixation and intercept (unchanged)
    fixation_regressors, fixation_betas_name = create_fixation_matrix(
        all_pupil_data[run]["eye_x"].values,
        all_pupil_data[run]["eye_y"].values,
    )
    intercept_regressor = np.ones((len(time_vector), 1))
    intercept_betas_name = [f"intercept-{dataset}_sub-{sub}_run-{run}"]

    design_matrix = np.hstack((
        onset_regressors, response_regressors,
        fixation_regressors, intercept_regressor,
        decision_regressors,
    ))
    betas_name = (
        onset_betas_name + response_betas_name
        + fixation_betas_name + intercept_betas_name
        + decision_betas_name
    )

    return design_matrix, betas_name


def collapse_basis_betas(results_run, basis):
    """Collapse per-component betas into a single signed-peak magnitude per trial.

    For each trial, reconstructs the predicted pupil response as:
        response(t) = sum_k  beta_k * basis_k(t)
    and uses the value at the time of maximum absolute deflection as the
    trial-level magnitude.  This preserves sign (dilation vs. constriction)
    while accounting for timing and dispersion variability captured by the
    non-canonical components.

    Applied to both decision and response regressors.  Onset and nuisance
    regressors are left untouched.

    Parameters
    ----------
    results_run : pd.DataFrame
        Columns: betas_estimates (float), betas_names (str), r2 (float).
        Contains one row per basis-expanded regressor.
    basis : np.ndarray, shape (T_basis, K)

    Returns
    -------
    pd.DataFrame with the same columns but with decision/response rows
    collapsed to one row per trial.
    """
    n_components = basis.shape[1]
    non_target_rows = results_run[
        ~results_run["betas_names"].str.contains("decision|response")
    ].copy()
    collapsed_parts = [non_target_rows]

    for regressor_type in ("decision", "response"):
        mask = results_run["betas_names"].str.contains(regressor_type)
        subset = results_run[mask].copy()

        # Strip _component-{k} suffix to recover the per-trial base label
        subset["trial_label"] = subset["betas_names"].str.replace(
            r"_component-\d+$", "", regex=True
        )

        collapsed_rows = []
        for label in subset["trial_label"].unique():
            trial_rows = (
                subset[subset["trial_label"] == label]
                .sort_values("betas_names")  # ensures component-0 < component-1 < component-2
                .reset_index(drop=True)
            )
            betas_k = trial_rows["betas_estimates"].values  # shape (K,)

            reconstructed = np.zeros(basis.shape[0])
            for k in range(n_components):
                reconstructed += betas_k[k] * basis[:, k]

            peak_idx = np.argmax(np.abs(reconstructed))
            magnitude = float(reconstructed[peak_idx])

            collapsed_rows.append({
                "betas_estimates": magnitude,
                "betas_names": label,
                "r2": trial_rows["r2"].iloc[0],
            })

        collapsed_parts.append(pd.DataFrame(collapsed_rows))

    return pd.concat(collapsed_parts, ignore_index=True)


def glm_single_run_basis(all_pupil_data, basis,
                          start_timings, response_timings,
                          sub, run, dataset):
    start_timings_run = start_timings.query("run == @run")
    response_timings_run = response_timings.query("run == @run")

    design_matrix, betas_name = build_design_matrix_run_basis(
        all_pupil_data, basis,
        start_timings_run, response_timings_run,
        sub, run, dataset,
    )

    y = all_pupil_data[run]["pupil"].values
    betas_estimates, residuals = ridge(design_matrix, y)

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"R-squared for sub-{sub} run-{run}: {r2:.4f}")

    results_run = pd.DataFrame({
        "betas_estimates": betas_estimates.tolist(),
        "betas_names": betas_name,
        "r2": [r2] * len(betas_name),
    })

    # Collapse multi-component betas into per-trial signed-peak magnitudes
    results_run = collapse_basis_betas(results_run, basis)

    vif_stats_decision = compute_vif(design_matrix, betas_name, sub, run, dataset, "decision")
    print(
        f"  Decision VIF — mean: {vif_stats_decision['mean_vif_decision']:.2f}, "
        f"sd: {vif_stats_decision['sd_vif_decision']:.2f} "
        f"(n={vif_stats_decision['n_regressors_decision']})"
    )
    vif_stats_response = compute_vif(design_matrix, betas_name, sub, run, dataset, "response")
    print(
        f"  Response VIF — mean: {vif_stats_response['mean_vif_response']:.2f}, "
        f"sd: {vif_stats_response['sd_vif_response']:.2f} "
        f"(n={vif_stats_response['n_regressors_response']})"
    )
    vif_stats = {**vif_stats_decision, **vif_stats_response}

    return results_run, vif_stats


def glm_single_sub_basis(all_pupil_data, basis,
                          start_timings, response_timings,
                          sub, runs, dataset):
    results_sub = pd.DataFrame()
    vif_records = []
    for run in runs:
        results_run, vif_stats = glm_single_run_basis(
            all_pupil_data, basis,
            start_timings, response_timings,
            sub, run, dataset,
        )
        results_run["run"] = run
        results_sub = pd.concat([results_sub, results_run], ignore_index=True)
        vif_records.append(vif_stats)
    results_sub["sub"] = sub
    results_sub["dataset"] = dataset
    return results_sub, pd.DataFrame(vif_records)
