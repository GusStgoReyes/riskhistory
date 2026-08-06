"""LOO model comparison aggregation for HDDM fits.

Loads per-model intermediate files produced by compute_diagnostics.py
and runs az.compare(). No large InferenceData files are loaded here.

Convergence diagnostics (rhat/ess) are collected for every model in
ALL_MODELS. LOO comparison and robustness checks are run separately for
each named group in MODEL_GROUPS, with output files named after the group.
"""
import argparse
import pickle
from pathlib import Path

import arviz as az
import pandas as pd
import numpy as np

# Every HDDM model that has been fit and diagnosed (see compute_diagnostics.py).
ALL_MODELS = [
    "HDDM_Base",
    "HDDM_Base_Lag1-stp",
    "HDDM_Base_Lag1-drift",
    "HDDM_Base_Lag1-stim",
    "HDDM_Base_Lag1-full",
    "HDDM_Base_Lag1-stpdrift",
    "HDDM_Base_Pupil",
    "HDDM_Base_Lag1-stpdrift_Pupil",
    "HDDM_Base_Lag1-stpdrift_Pupil_stp",
    "HDDM_Base_Lag1-stpdrift_Pupil_valuation",
    "HDDM_Base_Lag1-stpdrift_Pupil_drift",
    "HDDM_Base_Lag1-stpdrift_Pupil_boundary",
    "HDDM_Base_Lag1-stpdrift_Pupil_ndt",
]


MODEL_GROUPS = {
    "choicehistory_onemechanism" : ["HDDM_Base",
                                    "HDDM_Base_Lag1-stp",
                                    "HDDM_Base_Lag1-drift",
                                    "HDDM_Base_Lag1-stim"],

    "choicehistory_multimechanism" : ["HDDM_Base",
                                      "HDDM_Base_Lag1-full",
                                      "HDDM_Base_Lag1-stpdrift"],
    "arousal_choicehistory_bestmodel": ["HDDM_Base_Pupil",
                                      "HDDM_Base_Lag1-stpdrift_Pupil"],
    "arousal_mechanisms" : ["HDDM_Base_Lag1-stpdrift_Pupil",
                         "HDDM_Base_Lag1-stpdrift_Pupil_stp",
                         "HDDM_Base_Lag1-stpdrift_Pupil_valuation",
                         "HDDM_Base_Lag1-stpdrift_Pupil_drift",
                         "HDDM_Base_Lag1-stpdrift_Pupil_boundary",
                         "HDDM_Base_Lag1-stpdrift_Pupil_ndt",]
}


def _load_model_diagnostics(model_name: str, dataset: str, output_dir: Path):
    """Load the loo/rhat/ess intermediate files for one model."""
    loo_path = output_dir / f"{model_name}_{dataset}_loo.pkl"
    rhat_path = output_dir / f"{model_name}_{dataset}_rhat.csv"
    ess_path = output_dir / f"{model_name}_{dataset}_ess.csv"
    for p in (loo_path, rhat_path, ess_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run compute_diagnostics.py first")

    with open(loo_path, "rb") as f:
        loo = pickle.load(f)
    rhat = pd.read_csv(rhat_path)
    ess = pd.read_csv(ess_path)
    return loo, rhat, ess


def save_convergence_diagnostics(models: list[str], dataset: str, results_pth: str) -> None:
    """Collect rhat/ess across every model in `models` into one combined file each."""
    output_dir = Path(results_pth) / "hddm_compare"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rhat_frames = []
    ess_frames = []
    for model_name in models:
        _, rhat, ess = _load_model_diagnostics(model_name, dataset, output_dir)
        rhat_frames.append(rhat)
        ess_frames.append(ess)

    pd.concat(rhat_frames, ignore_index=True).to_csv(
        summary_dir / f"{dataset}_rhat.csv", index=False
    )
    pd.concat(ess_frames, ignore_index=True).to_csv(
        summary_dir / f"{dataset}_ess.csv", index=False
    )
    print(f"Convergence diagnostics (rhat/ess) for {len(models)} models saved to {summary_dir}",
          flush=True)


def run_group_comparison(group_name: str, models: list[str], dataset: str, results_pth: str) -> None:
    """Run LOO comparison + robustness check for one named group of models."""
    output_dir = Path(results_pth) / "hddm_compare"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    loo_dict: dict[str, az.ELPDData] = {}
    for model_name in models:
        loo, _, _ = _load_model_diagnostics(model_name, dataset, output_dir)
        loo_dict[model_name] = loo
    labels = list(loo_dict.keys())

    # --- Full comparison (all trials) ---
    print(f"[{group_name}] Running LOO comparison (all trials)...", flush=True)
    compare_df = az.compare(loo_dict, ic="loo")
    compare_df.index.name = "model"
    compare_df.reset_index().to_csv(
        summary_dir / f"{dataset}_{group_name}_model_comparison_loo.csv", index=False
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
        summary_dir / f"{dataset}_{group_name}_pareto_k_summary.csv", index=False
    )
    print(pd.DataFrame(k_summary).to_string(), flush=True)

    # --- Robustness: comparison excluding trials unreliable in ANY model ---
    try:
        ks = np.vstack([np.asarray(loo_dict[m].pareto_k.values) for m in labels])
        good = (ks <= 0.7).all(axis=0)   # reliable in every model
        n_dropped = int((~good).sum())
        print(f"[{group_name}] Robustness check: dropping {n_dropped} trials with k>0.7 in any model",
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
        rob_df.to_csv(summary_dir / f"{dataset}_{group_name}_robustness_loo.csv", index=False)
        print(rob_df.to_string(), flush=True)
    except AttributeError:
        print("WARNING: loo_i/pareto_k not available — recompute az.loo with "
              "pointwise=True to enable the robustness check.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate HDDM convergence diagnostics and run LOO comparisons per group"
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

    save_convergence_diagnostics(
        models=ALL_MODELS, dataset=args.dataset_name, results_pth=args.results_pth
    )

    if not MODEL_GROUPS:
        print("MODEL_GROUPS is empty — no comparisons to run. "
              "Fill in MODEL_GROUPS in compare_models.py to run comparisons.", flush=True)

    for group_name, models in MODEL_GROUPS.items():
        run_group_comparison(
            group_name=group_name,
            models=models,
            dataset=args.dataset_name,
            results_pth=args.results_pth,
        )


if __name__ == "__main__":
    main()
