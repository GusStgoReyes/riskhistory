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
    "equalIndifference": "equalIndifference",
    "equalRange": "equalRange",
    "sheng2020": "sheng2020",
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


def load_baseline_betas():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "1_baselinebehavior_betas.csv")
    return pd.read_csv(path)


def load_choicehistory_betas():
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / "2_choicehistory_betas.csv")
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


def load_pupilbins_choiceprev1_predicted():
    path = (REPO_ROOT / "results" / "analysis_pupilbehavior"
            / "2b_pupilbins_choiceprev1_predicted.csv")
    return pd.read_csv(path)


def load_pupilbins_choiceprev1_contrasts():
    path = (REPO_ROOT / "results" / "analysis_pupilbehavior"
            / "2b_pupilbins_choiceprev1_contrasts.csv")
    return pd.read_csv(path)


def load_model_comparison(comparison_type="behavior"):
    filenames = {
        "behavior": "2_behavior_model_comparison_2.csv",
    }
    path = (REPO_ROOT / "results" / "analysis_acceptglme"
            / filenames[comparison_type])
    return pd.read_csv(path)


def load_loo_comparison(dataset):
    path = (REPO_ROOT / "results" / "analysis_ddm" / "hddm_compare"
            / f"{dataset}_model_comparison_loo.csv")
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


def load_rtlme_interaction_emmeans():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "2_rtlme_interaction_emmeans.csv")
    return pd.read_csv(path)


def load_rtlme_interaction_contrasts():
    path = (REPO_ROOT / "results" / "analysis_rtlme"
            / "2_rtlme_interaction_contrasts.csv")
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
    path = _PPCDDM_DIR / f"{model_id}_simulated_data.csv"
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


def load_pupiltimeseries_emmeans(alignment: str = "start") -> pd.DataFrame:
    filenames = {
        "start":    "1_timeseries_start_emmeans.csv",
        "response": "1_timeseries_response_emmeans.csv",
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
            / "3_pupilbetas_transition_emmeans.csv")
    return pd.read_csv(path)


def load_pupilbetas_transition_contrasts():
    path = (REPO_ROOT / "results" / "analysis_pupilbetas"
            / "3_pupilbetas_transition_contrasts.csv")
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
