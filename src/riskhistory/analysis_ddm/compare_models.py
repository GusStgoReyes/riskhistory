"""WAIC-based model comparison and convergence diagnostics for HDDM fits."""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def load_inference_data(model_name: str, dataset: str, hddm_dir: Path) -> az.InferenceData:
    nc_path = hddm_dir / f"{model_name}_{dataset}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    return az.from_netcdf(str(nc_path))


def idata_to_tidy(ds, model_id: str, stat_name: str) -> pd.DataFrame:
    rows = []
    for var in ds.data_vars:
        values = np.atleast_1d(ds[var].values).flatten()
        if len(values) == 1:
            rows.append({"model": model_id, "variable": var, stat_name: values[0]})
        else:
            for i, v in enumerate(values):
                rows.append({"model": model_id, "variable": f"{var}[{i}]", stat_name: v})
    return pd.DataFrame(rows)


def run_comparison(
    models: list[str],
    dataset: str,
    results_pth: str,
) -> None:
    hddm_dir = Path(results_pth) / "hddm"
    output_dir = Path(results_pth) / "hddm_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process one model at a time to avoid holding all ~20 GB files in memory.
    # az.compare() accepts pre-computed ELPDData, so we free each InferenceData
    # before loading the next.
    waic_dict: dict[str, az.ELPDData] = {}
    rhat_frames = []
    ess_frames = []

    for model_name in models:
        print(f"Loading {model_name}_{dataset}.nc ...", flush=True)
        idata = load_inference_data(model_name, dataset, hddm_dir)

        print(f"  Computing WAIC ...", flush=True)
        waic_dict[model_name] = az.waic(idata, pointwise=True)

        print(f"  Computing rhat/ess ...", flush=True)
        rhat_frames.append(idata_to_tidy(az.rhat(idata), model_name, "rhat"))
        ess_frames.append(idata_to_tidy(az.ess(idata), model_name, "ess"))

        del idata  # free ~20 GB before loading next model

    # Model comparison
    print("Running WAIC comparison...", flush=True)
    compare_df = az.compare(waic_dict, ic="waic")
    compare_df.index.name = "model"
    compare_df.reset_index().to_csv(
        output_dir / f"{dataset}_model_comparison_waic.csv", index=False
    )
    print(compare_df.to_string(), flush=True)

    pd.concat(rhat_frames, ignore_index=True).to_csv(
        output_dir / f"{dataset}_rhat.csv", index=False
    )
    pd.concat(ess_frames, ignore_index=True).to_csv(
        output_dir / f"{dataset}_ess.csv", index=False
    )
    print(f"Results saved to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="WAIC model comparison and convergence diagnostics for HDDM fits"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model IDs to compare",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="equalIndifference",
        choices=["equalIndifference", "sheng2020", "equalRange"],
        help="Dataset name",
    )
    parser.add_argument(
        "--results_pth",
        type=str,
        required=True,
        help="Path to results directory (same as used during fitting)",
    )
    args = parser.parse_args()

    run_comparison(
        models=args.models,
        dataset=args.dataset_name,
        results_pth=args.results_pth,
    )


if __name__ == "__main__":
    main()
