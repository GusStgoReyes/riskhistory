"""LOO model comparison aggregation for HDDM fits.

Loads per-model intermediate files produced by compute_diagnostics.py
and runs az.compare(). No large InferenceData files are loaded here.
"""
import argparse
import pickle
from pathlib import Path

import arviz as az
import pandas as pd


def run_comparison(models: list[str], dataset: str, results_pth: str) -> None:
    output_dir = Path(results_pth) / "hddm_compare"

    loo_dict: dict[str, az.ELPDData] = {}
    rhat_frames = []
    ess_frames = []

    for model_name in models:
        loo_path = output_dir / f"{model_name}_{dataset}_loo.pkl"
        rhat_path = output_dir / f"{model_name}_{dataset}_rhat.csv"
        ess_path = output_dir / f"{model_name}_{dataset}_ess.csv"

        for p in (loo_path, rhat_path, ess_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found — run compute_diagnostics.py first"
                )

        with open(loo_path, "rb") as f:
            loo_dict[model_name] = pickle.load(f)

        rhat_frames.append(pd.read_csv(rhat_path))
        ess_frames.append(pd.read_csv(ess_path))

    print("Running LOO comparison...", flush=True)
    compare_df = az.compare(waic_dict, ic="loo")
    compare_df.index.name = "model"
    compare_df.reset_index().to_csv(
        output_dir / "summary" / f"{dataset}_model_comparison_loo.csv", index=False
    )
    print(compare_df.to_string(), flush=True)

    pd.concat(rhat_frames, ignore_index=True).to_csv(
        output_dir / "summary" / f"{dataset}_rhat.csv", index=False
    )
    pd.concat(ess_frames, ignore_index=True).to_csv(
        output_dir / "summary" / f"{dataset}_ess.csv", index=False
    )
    print(f"Results saved to {output_dir}/summary", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-model WAIC results and run model comparison"
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="List of model IDs to compare"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="equalIndifference",
        choices=["equalIndifference", "sheng2020", "equalRange"],
    )
    parser.add_argument(
        "--results_pth", type=str, required=True, help="Path to results directory"
    )
    args = parser.parse_args()

    run_comparison(
        models=args.models,
        dataset=args.dataset_name,
        results_pth=args.results_pth,
    )


if __name__ == "__main__":
    main()
