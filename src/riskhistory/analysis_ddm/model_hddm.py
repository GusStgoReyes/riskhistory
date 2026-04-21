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
    pupil_pth: str,
    results_pth: str,
    subj_IDs: list = None,
    idx_save: str = "",
    pupil: bool = False,
    burnin: int = 2000,
    sampling_iters: int = 10000,
    chains: int = 4,
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
    pupil_pth : str
        Path to the pupil betas CSV file.
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
        pupil_pth=pupil_pth,
        dataset_name=dataset_name,
        subj_IDs=subj_IDs,
        pupil=pupil
    )

    config = get_model_config(model_id)
    backend = config.get("backend", "HDDMnnRegressor")

    shared_kwargs = dict(
        informative=config["informative"],
        is_group_model=config["is_group_model"],
        group_only_regressors=config["group_only_regressors"],
        group_only_nodes=config["group_only_nodes"],
        include=config["include"],
    )

    if backend == "HDDMnnRegressor":
        m_reg = hddm.HDDMnnRegressor(
            data,
            config["reg_descr"],
            model=config["model"],
            **shared_kwargs,
        )
        m_reg.sample(
            sampling_iters,
            burn=burnin,
            chains=chains,
            loglike=True,
            return_infdata=True,
            save_name=f"{results_pth}/hddm/{model_id}_{dataset_name}{idx_save}",
        )
    elif backend == "HDDMRegressor":
        m_reg = hddm.HDDMRegressor(
            data,
            config["reg_descr"],
            **shared_kwargs,
        )
        m_reg.sample(
            sampling_iters,
            burn=burnin,
            chains=chains,
            loglike=False,
            return_infdata=True,
            save_name=f"{results_pth}/hddm/{model_id}_{dataset_name}{idx_save}",
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Expected 'HDDMnnRegressor' or 'HDDMRegressor'.")

    stats = m_reg.gen_stats()
    stats = stats.reset_index().rename(columns={"index": "param_name"})
    print(stats, flush=True)

    save_model_statistics(m_reg, stats, model_id, dataset_name, results_pth, idx_save)
    save_subj_statistics(stats, model_id, dataset_name, results_pth, idx_save)


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
        "--pupil_pth",
        type=str,
        required=True,
        help="Path to pupil betas CSV file",
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
        chains = 4
    else:
        subj_IDs = None
        idx_save = ""
        burnin = 2000
        sampling_iters = 10000
        chains = 4

    print(f"Fitting model {args.model_id} to dataset {args.dataset_name}...", flush=True)
    pupil = "Pupil" in args.model_id
    
    fit_model(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        data_pth=args.data_pth,
        pupil_pth=args.pupil_pth,
        results_pth=args.results_pth,
        subj_IDs=subj_IDs,
        idx_save=idx_save,
        pupil=pupil,
        burnin=burnin,
        sampling_iters=sampling_iters,
        chains=chains
    )


if __name__ == "__main__":
    main()
