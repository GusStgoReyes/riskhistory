"""Statistics and results saving utilities for HDDM fitting."""
import numpy as np
import pandas as pd
from pathlib import Path


def aic(model) -> float:
    """Calculate Akaike Information Criterion for model.

    Parameters
    ----------
    model : hddm.HDDMnnRegressor
        Fitted HDDM model.

    Returns
    -------
    float
        AIC value.
    """
    k = len(model.get_stochastics())
    logp = sum([x.logp for x in model.get_observeds()["node"]])
    return 2 * k - 2 * logp


def bic(model) -> float:
    """Calculate Bayesian Information Criterion for model.

    Parameters
    ----------
    model : hddm.HDDMnnRegressor
        Fitted HDDM model.

    Returns
    -------
    float
        BIC value.
    """
    k = len(model.get_stochastics())
    n = len(model.data)
    logp = sum([x.logp for x in model.get_observeds()["node"]])
    return -2 * logp + k * np.log(n)


def save_model_statistics(
    model,
    stats: pd.DataFrame,
    model_id: str,
    dataset_name: str,
    results_pth: str,
    idx_save: str = "",
) -> None:
    """Save group-level model parameters with fit statistics.

    Parameters
    ----------
    model : hddm.HDDMnnRegressor
        Fitted HDDM model.
    stats : pd.DataFrame
        Statistics DataFrame from model.gen_stats().
    model_id : str
        Model identifier.
    dataset_name : str
        Dataset name.
    results_pth : str
        Path to results directory.
    idx_save : str, optional
        Suffix for output filename.
    """
    output_dir = Path(results_pth) / "hddm_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_model = stats[~stats["param_name"].str.contains("subj")]
    df_model = df_model[["param_name", "mean", "std"]]
    df_model = df_model.rename(columns={"mean": "param_value", "std": "param_std"})
    df_model["model"] = model_id
    df_model["dataset"] = dataset_name
    df_model["dic"] = model.dic
    df_model["aic"] = aic(model)
    df_model["bic"] = bic(model)

    output_path = output_dir / f"{model_id}_{dataset_name}{idx_save}_model_stats.csv"
    df_model.to_csv(output_path, index=False)


def save_subj_statistics(
    stats: pd.DataFrame,
    model_id: str,
    dataset_name: str,
    results_pth: str,
    idx_save: str = "",
) -> None:
    """Extract and save subject-specific parameters from posterior.

    Parameters
    ----------
    stats : pd.DataFrame
        Statistics DataFrame from model.gen_stats().
    model_id : str
        Model identifier.
    dataset_name : str
        Dataset name.
    results_pth : str
        Path to results directory.
    idx_save : str, optional
        Suffix for output filename.
    """
    output_dir = Path(results_pth) / "hddm_params"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_subj = stats[stats["param_name"].str.contains("subj")]
    df_subj = df_subj[["param_name", "mean", "std"]]
    df_subj = df_subj.rename(columns={"mean": "param_value", "std": "param_std"})
    df_subj["sub"] = df_subj["param_name"].str.extract(r"subj\.(\d+)").astype(int)
    df_subj["param_name"] = df_subj["param_name"].str.replace(r"_subj\.\d+", "", regex=True)
    df_subj["model"] = model_id
    df_subj["dataset"] = dataset_name

    output_path = output_dir / f"{model_id}_{dataset_name}{idx_save}_subj_params.csv"
    df_subj.to_csv(output_path, index=False)


def ensure_output_dirs(results_pth: str) -> None:
    """Ensure all required output directories exist.

    Parameters
    ----------
    results_pth : str
        Base results directory path.
    """
    base = Path(results_pth)
    for subdir in ["db", "hddm", "hddm_stats", "hddm_params"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
