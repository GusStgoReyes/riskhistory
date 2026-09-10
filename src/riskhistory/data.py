"""Data loaders and dataset constants for figure generation."""

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DATASETS = ["equalIndifference", "equalRange", "sheng2020"]

DATASET_RUNS = {
    "equalIndifference": [1, 2, 3, 4],
    "equalRange":        [1, 2, 3, 4],
    "sheng2020":         [1, 2],
}

DATASET_LABELS = {
    "equalIndifference": "EqualIndifference",
    "equalRange": "EqualRange",
    "sheng2020": "Sheng2020",
}

DATASET_CURRENCY = {
    "equalIndifference": "ILS",
    "equalRange": "ILS",
    "sheng2020": "USD",
}

GAIN_CENTERS = {
    "equalIndifference": 25.0,
    "equalRange": 12.5,
    "sheng2020": 5.5,
}
LOSS_CENTERS = {
    "equalIndifference": 12.5,
    "equalRange": 12.5,
    "sheng2020": 5.5,
}

LOSS_AVERSION = {
    "equalIndifference": 1.3297,
    "equalRange": 1.0262,
    "sheng2020": 1.6585,
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_behavioral_data(exclude=True):
    path = REPO_ROOT / "data" / "preprocessed" / "behavioral_data.csv"
    df = pd.read_csv(path)
    if exclude:
        df = df[df["exclude"] != True].copy()
    return df


def load_behavioral_data_pupilclean(alignment="start", exclude=True):
    """Behavioral data restricted to trials that also pass pupil exclusion.

    Applies the standard behavioral exclusion (load_behavioral_data()'s
    default) first, then filters to trials with exclude_{alignment}Aligned
    == False in pupil_data_statistics_with_exclusions.csv -- the same
    trial-level exclusion decision already reflected in
    pupil_data_alignedStartTime.csv / pupil_data_alignedResponseTime.csv's
    own per-timebin exclude column, just at trial grain instead of
    per-timebin.

    alignment : "start" or "response"
    """
    df = load_behavioral_data(exclude=exclude)

    stats_path = (REPO_ROOT / "results" / "setup_pupildata"
                  / "pupil_data_statistics_with_exclusions.csv")
    stats = pd.read_csv(stats_path)

    excl_col = f"exclude_{alignment}Aligned"
    stats[excl_col] = stats[excl_col].astype(str).str.lower() == "true"
    stats = stats[["sub", "dataset", "trial", excl_col]].drop_duplicates()

    merged = df.merge(stats, on=["sub", "dataset", "trial"], how="inner")
    return merged[merged[excl_col] == False].drop(columns=[excl_col])


def load_baseline_betas():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "1_baselinebehavior_betas.csv")
    return pd.read_csv(path)


def load_choicehistory_betas():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "2_choicehistory_betas.csv")
    return pd.read_csv(path)


def load_choicehistorypersubject_betas():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "4_choicehistorypersubject_betas.csv")
    return pd.read_csv(path)


def load_choicehistory_permutation():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "5_choicehistory_permutation.csv")
    return pd.read_csv(path)


def load_rtgam_fitted_curves():
    path = (REPO_ROOT / "results" / "analysis_rtgam"
            / "1_rtgam_fitted_curves.csv")
    return pd.read_csv(path)


def load_rtgam_parametric_effects():
    path = (REPO_ROOT / "results" / "analysis_rtgam"
            / "1_rtgam_parametric_effects.csv")
    return pd.read_csv(path)


def load_rtgam_smooth_effects():
    path = (REPO_ROOT / "results" / "analysis_rtgam"
            / "1_rtgam_smooth_effects.csv")
    return pd.read_csv(path)


def load_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "2_behavior_fixed_effects.csv")
    return pd.read_csv(path)


def load_choicehistorysvpartitions_predictions():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "3_choicehistory_svpartitions_predictions.csv")
    return pd.read_csv(path)


def load_choicehistorysvpartitions_contrasts():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "3_choicehistory_svpartitions_contrasts.csv")
    return pd.read_csv(path)


def load_sv_partition_means(n_partitions=5):
    """Mean subjective value (sv) within each equal-count sv partition, per
    dataset -- gives representative sv values for the partitions used by
    3_choicehistory_svpartitions_contrasts.csv (sv_partition 1..n_partitions).

    Mirrors the R pipeline's sv computation (add_subjectivevalues) and
    partitioning (standardize_behavioral_regressors's sv_partition) in
    utils_createregressors.R, computed here only to label the partitions --
    the choice-history contrasts themselves still come from
    load_choicehistorysvpartitions_contrasts().

    Two-step averaging, matching the R workflow: mean sv per sub/dataset/
    partition first, then mean of those subject-level means per
    dataset/partition.
    """
    df = load_behavioral_data(exclude=False)
    betas = load_baseline_betas()

    df = df.merge(betas, on=["sub", "dataset"])
    df["sv"] = (
        df["beta_intercept"] / df["beta_gain"] + df["gain_centered"]
        + df["beta_loss"] / df["beta_gain"] * df["loss_centered"]
    )

    df = df[df["exclude"] != True].copy()

    df["sv_partition"] = df.groupby(["dataset", "sub"])["sv"].transform(
        lambda x: pd.qcut(x, n_partitions, labels=False) + 1
    )

    persub_means = (
        df.groupby(["dataset", "sub", "sv_partition"])["sv"]
        .mean().reset_index()
    )
    return (
        persub_means.groupby(["dataset", "sv_partition"])["sv"]
        .mean().reset_index()
    )


def load_pupilacceptmodel_historyeffect_by_pupilbin():
    path = (REPO_ROOT / "results" / "analysis_pupilbehavior"
            / "3_pupilacceptmodel_historyeffect_by_pupilbin.csv")
    return pd.read_csv(path)


def load_pupilacceptmodel_pupilbin_contrast():
    path = (REPO_ROOT / "results" / "analysis_pupilbehavior"
            / "3_pupilacceptmodel_pupilbin_contrast.csv")
    return pd.read_csv(path)


def load_pupilacceptmodel_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_pupilbehavior"
            / "3_pupilacceptmodel_fixed_effects.csv")
    return pd.read_csv(path)


def load_ddmppcbehavior_pupilaccept_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_ddmppcbehavior"
            / "3_pupilaccept_fixed_effects.csv")
    return pd.read_csv(path)


def load_ddmppcbehavior_pupilaccept_comparison_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_ddmppcbehavior"
            / "4_pupilaccept_comparison_fixed_effects.csv")
    return pd.read_csv(path)


def load_model_comparison(comparison_type="behavior"):
    filenames = {
        "behavior": "2_behavior_model_comparison.csv",
    }
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / filenames[comparison_type])
    return pd.read_csv(path)


def load_loo_comparison(dataset, group_name=None):
    """Pairwise LOO model comparison table for one dataset (see
    compare_models.py's compare_pair/run_group_comparison).

    One row per (reference_model, target_model) pair listed under that
    MODEL_GROUPS entry, with:
      - diff_elpd_loo: sum(elpd_target) - sum(elpd_reference) over all
        trials shared by the two models
      - diff_elpd_reliable: same diff, restricted to trials reliable
        (pareto_k <= 0.7) in both models
      - num_unreliable / num_total: trials unreliable in either model,
        out of the shared total

    group_name : str, optional
        Name of the MODEL_GROUPS entry in compare_models.py this comparison
        was run for (e.g. "arousal") -- must match exactly, since it's part
        of the output filename. Omit for the legacy, ungrouped comparison
        file.
    """
    suffix = f"_{group_name}" if group_name else ""
    path = (REPO_ROOT / "results" / "analysis_ddm" / "hddm_compare"
            / f"{dataset}{suffix}_model_comparison_loo.csv")
    return pd.read_csv(path)


def load_ddm_rhat(dataset):
    path = (REPO_ROOT / "results" / "analysis_ddm" / "hddm_compare"
            / f"{dataset}_rhat.csv")
    return pd.read_csv(path)


def load_ddm_ess(dataset):
    path = (REPO_ROOT / "results" / "analysis_ddm" / "hddm_compare"
            / f"{dataset}_ess.csv")
    return pd.read_csv(path)


def load_eta_profile():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "3_choicehistoryprofile.csv")
    return pd.read_csv(path)


def load_rtlme_baseline_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "1_rtlme_baseline_fixed_effects.csv")
    return pd.read_csv(path)


def load_rtlme_interaction_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "2_rtlme_interaction_fixed_effects.csv")
    return pd.read_csv(path)


def load_rtlme_interaction_avgpred():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "2_rtlme_interaction_avgpred.csv")
    return pd.read_csv(path)


def load_rtlme_interaction_contrasts():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "2_rtlme_interaction_contrasts.csv")
    return pd.read_csv(path)


def load_rtlme_switchpersubject_betas():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "3_rtlme_switchpersubject_betas.csv")
    return pd.read_csv(path)


def load_convergence():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "4_model_convergence_report.csv")
    return pd.read_csv(path)


def load_pupilbetas():
    path = (REPO_ROOT / "results" / "setup_pupilbetas"
            / "pupilbetas_glm_results.csv")
    return pd.read_csv(path)


def load_pupil_data_raw(alignment="start"):
    """Per-timebin raw pupil timeseries (not baseline-corrected, not
    excluded): sub, trial, dataset, timebin, pupil_size, baseline_pupil,
    exclude."""
    filenames = {
        "start":    "pupil_data_alignedStartTime.csv",
        "response": "pupil_data_alignedResponseTime.csv",
    }
    path = REPO_ROOT / "data" / "preprocessed" / filenames[alignment]
    return pd.read_csv(path, low_memory=False)


_DDM_STATS_DIR      = REPO_ROOT / "results" / "analysis_ddm" / "hddm_stats"
_DDM_PARAMS_DIR     = REPO_ROOT / "results" / "analysis_ddm" / "hddm_params"
_DDM_POSTERIORS_DIR = REPO_ROOT / "results" / "analysis_ddm" / "hddm_posteriors"
_PPCDDM_DIR         = REPO_ROOT / "results" / "analysis_ppcddm"


def _ddm_idx_save(pupil_col: str) -> str:
    """Filename suffix for a given pupil variant -- must match model_hddm.py /
    utils_stats.py / compute_diagnostics.py / extract_posteriors.py exactly."""
    return "" if pupil_col == "pupil" else f"_{pupil_col}"


def load_ddm_model_stats(model_id: str, dataset: str, pupil_col: str = "pupil") -> pd.DataFrame:
    idx_save = _ddm_idx_save(pupil_col)
    path = _DDM_STATS_DIR / f"{model_id}_{dataset}{idx_save}_model_stats.csv"
    return pd.read_csv(path)


def load_ddm_subj_params(model_id: str, dataset: str, pupil_col: str = "pupil") -> pd.DataFrame:
    idx_save = _ddm_idx_save(pupil_col)
    path = _DDM_PARAMS_DIR / f"{model_id}_{dataset}{idx_save}_subj_params.csv"
    return pd.read_csv(path)


def load_ddm_simulated(model_id: str) -> pd.DataFrame:
    path = _PPCDDM_DIR / f"{model_id}_simulated_data_sampled.csv"
    return pd.read_csv(path)


def load_ddm_posterior(model_id: str, dataset: str, pupil_col: str = "pupil"):
    """MCMC posterior samples (group-level + per-subject) for a DDM fit.

    Returns the raw NpzFile; group-level keys are the bare parameter name
    (e.g. "v_choice_prev1", shape (n_chains, n_draws)), per-subject keys are
    suffixed "_subj.{sub}", and group SDs are suffixed "_std".

    pupil_col : "pupil" (raw GLM amplitude) or "pupil_residual" -- must match
    whichever pupil variant the model was fit with.
    """
    idx_save = _ddm_idx_save(pupil_col)
    path = _DDM_POSTERIORS_DIR / f"{model_id}_{dataset}{idx_save}_posterior.npz"
    return np.load(path, allow_pickle=True)


def load_pupiltimeseries_fixed_effects(alignment: str = "start") -> pd.DataFrame:
    filenames = {
        "start":    "1_timeseries_start_fixed_effects.csv",
        "response": "1_timeseries_response_fixed_effects.csv",
    }
    path = (REPO_ROOT / "results" / "analysis_pupiltimeseries" / filenames[alignment])
    return pd.read_csv(path)


def load_pupiltimeseries_avgpred(alignment: str = "start") -> pd.DataFrame:
    filenames = {
        "start":    "1_timeseries_start_avgpred.csv",
        "response": "1_timeseries_response_avgpred.csv",
    }
    path = (REPO_ROOT / "results" / "analysis_pupiltimeseries" / filenames[alignment])
    return pd.read_csv(path)


def load_pupiltimeseries_contrasts(alignment: str = "start") -> pd.DataFrame:
    filenames = {
        "start":    "1_timeseries_start_contrasts.csv",
        "response": "1_timeseries_response_contrasts.csv",
    }
    path = (REPO_ROOT / "results" / "analysis_pupiltimeseries" / filenames[alignment])
    return pd.read_csv(path)


def load_pupilbetas_fullmodel_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "2_pupilbetas_fullmodel_fixed_effects.csv")
    return pd.read_csv(path)


def load_pupilbetas_fullmodel_residuals():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "2_pupilbetas_fullmodel_residuals.csv")
    return pd.read_csv(path)


def load_pupilbetas_transition_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "3_pupilbetas_transition_fixed_effects.csv")
    return pd.read_csv(path)


def load_pupilbetas_transition_emmeans():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "2_pupilbetas_transition_emmeans.csv")
    return pd.read_csv(path)


def load_pupilbetas_transition_contrasts():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "2_pupilbetas_transition_contrasts.csv")
    return pd.read_csv(path)


def load_pupilbetas_choicealteration_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "3_pupilbetas_choicealteration_fixed_effects.csv")
    return pd.read_csv(path)


def load_pupilbetas_choicealteration_avgpred():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "3_pupilbetas_choicealteration_avgpred.csv")
    return pd.read_csv(path)


def load_pupilbetas_choicealteration_contrasts():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "3_pupilbetas_choicealteration_contrasts.csv")
    return pd.read_csv(path)


def load_pupilchoicehistory_switch_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_pupilchoicehistory"
            / "1_pupil_switch_ch_fixed_effects.csv")
    return pd.read_csv(path)


def load_pupilchoicehistory_accept_data():
    """Per-subject mean acceptance rate and pupil beta by transition
    (AA/AR/RR/RA), one row per (sub, dataset). See
    analysis_pupilchoicehistory/2_pupil_accept.R."""
    path = (REPO_ROOT / "results" / "analysis_pupilchoicehistory"
            / "2_pupil_accept_data.csv")
    return pd.read_csv(path)


def load_pupilchoicehistory_accept_fixed_effects():
    path = (REPO_ROOT / "results" / "analysis_pupilchoicehistory"
            / "2_pupil_accept_fixed_effects.csv")
    return pd.read_csv(path)


def load_pupilbetas_correlation():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "2_regressors_correlation.csv")
    return pd.read_csv(path)


def load_pupilbetas_lrt():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "1_pupilbetas_lrt.csv")
    return pd.read_csv(path)


def load_pupilbetas_preprocessed(exclude=True):
    path = (REPO_ROOT / "data" / "preprocessed"
            / "pupilbetas_glm_preprocessed.csv")
    df = pd.read_csv(path)
    df["exclude_startAligned"] = (
        df["exclude_startAligned"].astype(str).str.lower() == "true"
    )
    if exclude:
        df = df[~df["exclude_startAligned"]].copy()
    return df


def load_pupilbaseline_preprocessed(exclude=True):
    """Per-trial pre-stimulus pupil baseline (z-scored within dataset/sub/run).

    See src/riskhistory/setup_pupilbetas/run_setup_pupilbaseline.py.
    """
    path = (REPO_ROOT / "results" / "setup_pupilbetas"
            / "pupilbaseline_preprocessed.csv")
    df = pd.read_csv(path)
    df["exclude_startAligned"] = (
        df["exclude_startAligned"].astype(str).str.lower() == "true"
    )
    if exclude:
        df = df[~df["exclude_startAligned"]].copy()
    return df


def add_subjective_values(df, betas=None):
    """Merge betas and compute SV = intercept + gain*gain_c + loss*loss_c."""
    if betas is None:
        betas = load_baseline_betas()
    df = df.merge(betas, on=["sub", "dataset"], how="left")
    df["sv"] = (
        df["beta_intercept"]
        + df["beta_gain"] * df["gain_centered"]
        + df["beta_loss"] * df["loss_centered"]
    )
    return df
