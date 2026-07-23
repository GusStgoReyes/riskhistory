"""LOO model comparison aggregation for HDDM fits.

Loads per-model intermediate files produced by compute_diagnostics.py
and runs az.compare(). No large InferenceData files are loaded here.
"""
import argparse
import pickle
from pathlib import Path

import arviz as az
import pandas as pd
import numpy as np

def run_comparison(
    models: list[str],
    dataset: str,
    results_pth: str,
    pupil_cols: list[str] = None,
) -> None:
    output_dir = Path(results_pth) / "hddm_compare"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if pupil_cols is None:
        pupil_cols = ["pupil"] * len(models)

    loo_dict: dict[str, az.ELPDData] = {}
    rhat_frames = []
    ess_frames = []
    labels = []

    for model_name, pupil_col in zip(models, pupil_cols):
        # Must match compute_diagnostics.py's idx_save suffix exactly.
        idx_save = "" if pupil_col == "pupil" else f"_{pupil_col}"
        label = f"{model_name}{idx_save}"
        labels.append(label)

        loo_path = output_dir / f"{model_name}_{dataset}{idx_save}_loo.pkl"
        rhat_path = output_dir / f"{model_name}_{dataset}{idx_save}_rhat.csv"
        ess_path = output_dir / f"{model_name}_{dataset}{idx_save}_ess.csv"
        for p in (loo_path, rhat_path, ess_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found — run compute_diagnostics.py first")
        with open(loo_path, "rb") as f:
            loo_dict[label] = pickle.load(f)
        rhat_frames.append(pd.read_csv(rhat_path))
        ess_frames.append(pd.read_csv(ess_path))

    # --- Full comparison (all trials) ---
    print("Running LOO comparison (all trials)...", flush=True)
    compare_df = az.compare(loo_dict, ic="loo")   # FIX: loo_dict, not waic_dict
    compare_df.index.name = "model"
    compare_df.reset_index().to_csv(
        summary_dir / f"{dataset}_model_comparison_loo.csv", index=False
    )
    print(compare_df.to_string(), flush=True)

    # --- Pareto-k diagnostic per model ---
    k_summary = []
    for name, loo in loo_dict.items():
        k = np.asarray(loo.pareto_k.values)
        k_summary.append({
            "model": name,
            "n_obs": k.size,
            "n_k_gt_0.7": int((k > 0.7).sum()),
            "n_k_gt_1.0": int((k > 1.0).sum()),
            "n_k_inf": int(np.isinf(k).sum()),
            "max_k": float(k.max()),
        })
    pd.DataFrame(k_summary).to_csv(
        summary_dir / f"{dataset}_pareto_k_summary.csv", index=False
    )
    print(pd.DataFrame(k_summary).to_string(), flush=True)

    # --- Robustness: comparison excluding trials unreliable in ANY model ---
    try:
        ks = np.vstack([np.asarray(loo_dict[m].pareto_k.values) for m in labels])
        good = (ks <= 0.7).all(axis=0)   # reliable in every model
        n_dropped = int((~good).sum())
        print(f"Robustness check: dropping {n_dropped} trials with k>0.7 in any model",
              flush=True)

        rob_rows = []
        for m in labels:
            loo_i = np.asarray(loo_dict[m].loo_i.values).ravel()
            rob_rows.append({
                "model": m,
                "elpd_loo_all": float(loo_i.sum()),
                "elpd_loo_reliable_only": float(loo_i[good].sum()),
            })
        rob_df = pd.DataFrame(rob_rows).sort_values(
            "elpd_loo_reliable_only", ascending=False
        )
        rob_df.to_csv(summary_dir / f"{dataset}_robustness_loo.csv", index=False)
        print(rob_df.to_string(), flush=True)
    except AttributeError:
        print("WARNING: loo_i/pareto_k not available — recompute az.loo with "
              "pointwise=True to enable the robustness check.", flush=True)

    # --- diagnostics ---
    pd.concat(rhat_frames, ignore_index=True).to_csv(
        summary_dir / f"{dataset}_rhat.csv", index=False)
    pd.concat(ess_frames, ignore_index=True).to_csv(
        summary_dir / f"{dataset}_ess.csv", index=False)
    print(f"Results saved to {summary_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-model LOO results and run model comparison"
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="List of model IDs to compare"
    )
    parser.add_argument(
        "--pupil_cols",
        nargs="+",
        default=None,
        help="Pupil column used for each entry in --models, same length/order "
             "(defaults to 'pupil' for all if omitted). Only affects the "
             "filename suffix used to locate that model's diagnostics.",
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
        pupil_cols=args.pupil_cols,
    )


if __name__ == "__main__":
    main()
