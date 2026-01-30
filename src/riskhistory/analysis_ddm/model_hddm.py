"""HDDM model fitting for choice history analysis."""
import argparse

import hddm

from .utils_data import load_data
from .utils_model import get_model_config, get_available_models
from .utils_stats import (
    save_model_statistics,
    save_subj_statistics,
    ensure_output_dirs,
)


def fit_model(
    model_id: str,
    dataset_name: str,
    data_pth: str,
    results_pth: str,
    subj_IDs: list = None,
    idx_save: str = "",
    burnin: int = 2000,
    sampling_iters: int = 10000,
) -> None:
    """Fit an HDDM model to behavioral data.

    Parameters
    ----------
    model_id : str
        Model identifier (see utils_model.MODEL_REGISTRY for available models).
    dataset_name : str
        Name of the dataset ('equalIndifference', 'sheng2020', 'equalRange').
    data_pth : str
        Path to the behavioral CSV file.
    results_pth : str
        Path to save results.
    subj_IDs : list, optional
        List of subject IDs to include. If None, all subjects are used.
    idx_save : str, optional
        Suffix for output filenames.
    burnin : int, optional
        Number of burn-in samples (default: 2000).
    sampling_iters : int, optional
        Number of sampling iterations (default: 10000).
    """
    ensure_output_dirs(results_pth)

    data = load_data(
        data_pth=data_pth,
        dataset_name=dataset_name,
        subj_IDs=subj_IDs,
    )

    config = get_model_config(model_id)

    m_reg = hddm.HDDMnnRegressor(
        data,
        config["reg_descr"],
        model=config["model"],
        informative=config["informative"],
        is_group_model=config["is_group_model"],
        group_only_regressors=config["group_only_regressors"],
        group_only_nodes=config["group_only_nodes"],
        include=config["include"],
    )

    m_reg.sample(
        sampling_iters,
        burn=burnin,
        loglike=True,
        dbname=f"{results_pth}/db/{model_id}_{dataset_name}{idx_save}.db",
        db="pickle",
    )

    stats = m_reg.gen_stats()
    stats = stats.reset_index().rename(columns={"index": "param_name"})
    print(stats, flush=True)

    save_model_statistics(m_reg, stats, model_id, dataset_name, results_pth, idx_save)
    save_subj_statistics(stats, model_id, dataset_name, results_pth, idx_save)
    m_reg.save(f"{results_pth}/hddm/{model_id}_{dataset_name}{idx_save}.hddm")


def main():
    """CLI entry point for HDDM model fitting."""
    parser = argparse.ArgumentParser(
        description="Fit an HDDM model to behavioral data"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="HDDM_Collapse",
        choices=get_available_models(),
        help="Model ID to fit",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="equalIndifference",
        choices=["equalIndifference", "sheng2020", "equalRange"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data_pth",
        type=str,
        required=True,
        help="Path to behavioral CSV file",
    )
    parser.add_argument(
        "--results_pth",
        type=str,
        required=True,
        help="Path to save results",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test fit with fewer iterations",
    )
    args = parser.parse_args()

    if args.test:
        subj_IDs = [1, 3, 5, 9]
        idx_save = "_test"
        burnin = 2
        sampling_iters = 10
    else:
        subj_IDs = None
        idx_save = ""
        burnin = 2000
        sampling_iters = 10000

    print(f"Fitting model {args.model_id} to dataset {args.dataset_name}...", flush=True)

    fit_model(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        data_pth=args.data_pth,
        results_pth=args.results_pth,
        subj_IDs=subj_IDs,
        idx_save=idx_save,
        burnin=burnin,
        sampling_iters=sampling_iters,
    )


if __name__ == "__main__":
    main()
