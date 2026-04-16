"""Statistics and results saving utilities for HDDM fitting."""
import numpy as np
import pandas as pd
from pathlib import Path

import arviz as az
import xarray as xr


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

def _patch_infdata_loglik(nc_path, m_reg, chains):
    """Inject log_likelihood into a saved InferenceData if it is missing.

    HDDM computes lppd but two dtype bugs prevent it from being written to
    the .nc file:
      - lppd is float32; netCDF4 expects float64 ('double')
      - some InferenceData variables are bool, which netCDF4 cannot store

    This function loads the saved .nc, adds the log_likelihood group from the
    in-memory m_reg.lppd (cast to float64), casts all other groups to float64
    where applicable, and re-saves.

    Parameters
    ----------
    nc_path : str
        Path to the saved .nc InferenceData file.
    m_reg : hddm.HDDMnnRegressor
        Fitted model still in memory (lppd attribute must be present).
    chains : int
        Number of chains used during sampling.
    """
    if not hasattr(m_reg, "lppd") or m_reg.lppd is None:
        print("Warning: m_reg.lppd not available; log_likelihood group will be missing.")
        return

    try:
        idata = az.from_netcdf(nc_path)
    except Exception as e:
        print(f"Warning: could not load InferenceData from {nc_path}: {e}")
        return

    # Cast all existing groups to float64 to avoid dtype issues on reload
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        for var in list(group.data_vars):
            if group[var].dtype in (np.float32,):
                group[var] = group[var].astype(np.float64)
            elif group[var].dtype == bool:
                group[var] = group[var].astype(np.int8)

    # Build log_likelihood group from lppd
    # lppd shape is (n_total_samples, n_observations) — split into (chains, draws, obs)
    lppd = np.array(m_reg.lppd, dtype=np.float64)
    n_total, n_obs = lppd.shape
    n_draws = n_total // chains
    lppd_reshaped = lppd[: chains * n_draws].reshape(chains, n_draws, n_obs)

    log_lik_ds = xr.Dataset(
        {"y": xr.DataArray(lppd_reshaped, dims=["chain", "draw", "obs"])}
    )
    idata.add_groups({"log_likelihood": log_lik_ds})
    idata.to_netcdf(nc_path)
    print(f"Patched log_likelihood into {nc_path} (shape: {lppd_reshaped.shape})")
