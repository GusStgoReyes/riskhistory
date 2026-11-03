"""LOO model comparison aggregation for HDDM fits.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Every HDDM model that has been fit and diagnosed (see compute_diagnostics.py).
ALL_MODELS = [
    "HDDM_Base",
    "HDDM_Base_Lag1-stp",
    "HDDM_Base_Lag1-drift",
    "HDDM_Base_Lag1-stim",
    "HDDM_Base_Lag1-full",
    "HDDM_Base_Lag1-stpdrift",
    "HDDM_Base_Pupil",
    "HDDM_Base_Lag1-drift_Pupil",
    "HDDM_Base_Lag1-stp_Pupil",
    "HDDM_Base_Lag1-drift_Pupil_drift",
    "HDDM_Base_Lag1-stp_Pupil_stp",
    "HDDM_Base_Pupil_valuation",
    "HDDM_Base_Lag1-stpdrift_Pupil",
    "HDDM_Base_Lag1-stpdrift_Pupil_stp",
    "HDDM_Base_Lag1-stpdrift_Pupil_drift",
    "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift",
    "HDDM_Base_Lag1-stpdrift_Pupil_driftmain",
    "HDDM_Base_Lag1-stpdrift_Pupil_stpmain",
]
    

# Each group is a list of [reference_model, target_model] pairs; the LOO
# comparison is a pairwise diff, not an all-at-once az.compare() ranking.
MODEL_GROUPS = {
    "choicehistory" : [["HDDM_Base", "HDDM_Base_Lag1-stp"],
                       ["HDDM_Base", "HDDM_Base_Lag1-drift"],
                       ["HDDM_Base", "HDDM_Base_Lag1-stim"],
                       ["HDDM_Base", "HDDM_Base_Lag1-stpdrift"],
                       ["HDDM_Base", "HDDM_Base_Lag1-full"],
                       ["HDDM_Base_Lag1-stp", "HDDM_Base_Lag1-stpdrift"],
                       ["HDDM_Base_Lag1-stp", "HDDM_Base_Lag1-full"],
                       ["HDDM_Base_Lag1-stpdrift", "HDDM_Base_Lag1-full"],

                       ],
    "arousal" : [["HDDM_Base_Pupil", "HDDM_Base_Lag1-drift_Pupil"],
                 ["HDDM_Base_Pupil", "HDDM_Base_Lag1-stp_Pupil"],
                 ["HDDM_Base_Lag1-drift_Pupil", "HDDM_Base_Lag1-drift_Pupil_drift"],
                 ["HDDM_Base_Lag1-stp_Pupil", "HDDM_Base_Lag1-stp_Pupil_stp"],
                 ["HDDM_Base_Pupil", "HDDM_Base_Pupil_valuation"],
                 ["HDDM_Base_Pupil", "HDDM_Base_Lag1-stpdrift_Pupil"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil", "HDDM_Base_Lag1-stpdrift_Pupil_stp"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil", "HDDM_Base_Lag1-stpdrift_Pupil_drift"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil", "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil", "HDDM_Base_Lag1-stpdrift_Pupil_driftmain"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil", "HDDM_Base_Lag1-stpdrift_Pupil_stpmain"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil_driftmain", "HDDM_Base_Lag1-stpdrift_Pupil_drift"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil_stpmain", "HDDM_Base_Lag1-stpdrift_Pupil_stp"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil_stp", "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift"],
                 ["HDDM_Base_Lag1-stpdrift_Pupil_drift", "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift"],
                ]
}


def _load_rhat_ess(model_name: str, dataset: str, output_dir: Path):
    """Load the rhat/ess intermediate files for one model."""
    rhat_path = output_dir / f"{model_name}_{dataset}_rhat.csv"
    ess_path = output_dir / f"{model_name}_{dataset}_ess.csv"
    for p in (rhat_path, ess_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run compute_diagnostics.py first")
    return pd.read_csv(rhat_path), pd.read_csv(ess_path)


def _load_loo_csv(model_name: str, dataset: str, output_dir: Path) -> pd.DataFrame:
    """Load one model's per-trial elpd/reliability table (see compute_diagnostics.py)."""
    loo_path = output_dir / f"{model_name}_{dataset}_loo.csv"
    if not loo_path.exists():
        raise FileNotFoundError(f"{loo_path} not found — run compute_diagnostics.py first")
    # nullable "boolean" dtype: reliable_k/bad_k are blank (NA) for bad_tail
    # trials, which pd.read_csv would otherwise silently read back as False.
    return pd.read_csv(loo_path, dtype={"reliable_k": "boolean", "bad_k": "boolean"})


def save_convergence_diagnostics(models: list[str], dataset: str, results_pth: str) -> None:
    """Collect rhat/ess across every model in `models` into one combined file each."""
    output_dir = Path(results_pth) / "hddm_compare"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rhat_frames = []
    ess_frames = []
    for model_name in models:
        rhat, ess = _load_rhat_ess(model_name, dataset, output_dir)
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


def compare_pair(
    reference_model: str, target_model: str, dataset: str, output_dir: Path
) -> dict:
    """LOO elpd diff between one target model and its reference, on their shared trials."""
    ref_df = _load_loo_csv(reference_model, dataset, output_dir)
    tgt_df = _load_loo_csv(target_model, dataset, output_dir)

    merged = ref_df.merge(
        tgt_df, on=["trial", "sub", "dataset"], suffixes=("_ref", "_tgt"), how="inner"
    )
    if len(merged) != len(ref_df) or len(merged) != len(tgt_df):
        print(
            f"WARNING: {reference_model} ({len(ref_df)} trials) and {target_model} "
            f"({len(tgt_df)} trials) don't share the same (sub, trial) set -- "
            f"comparing on {len(merged)} common trials.",
            flush=True,
        )

    # NA (bad_tail trials, which never got a pareto_k) counts as unreliable.
    reliable_ref = merged["reliable_k_ref"].fillna(False)
    reliable_tgt = merged["reliable_k_tgt"].fillna(False)
    common_reliable = reliable_ref & reliable_tgt

    # SE of the elpd diff clustered by subject: trials within a subject aren't
    # independent, so sum the pointwise diffs per subject and use
    # sqrt(n_subj * var(subject sums)). Unreliable trials contribute 0 to the
    # reliable sums so every subject is still counted.
    pointwise_diff = merged["elpd_tgt"] - merged["elpd_ref"]
    reliable_diff = pointwise_diff.where(common_reliable, 0.0)
    subj_keys = [merged["dataset"], merged["sub"]]
    subj_diff = pointwise_diff.groupby(subj_keys).sum()
    subj_reliable_diff = reliable_diff.groupby(subj_keys).sum()

    return {
        "reference_model": reference_model,
        "target_model": target_model,
        "diff_elpd_loo": float(pointwise_diff.sum()),
        "se_diff_elpd_loo": float(np.sqrt(len(subj_diff) * subj_diff.var())),
        "diff_elpd_reliable": float(reliable_diff.sum()),
        "se_diff_elpd_reliable": float(np.sqrt(len(subj_reliable_diff) * subj_reliable_diff.var())),
        "num_unreliable": int((~common_reliable).sum()),
        "num_total": len(merged),
    }


def run_group_comparison(
    group_name: str, pairs: list[list[str]], dataset: str, results_pth: str
) -> None:
    """Compute the reference-vs-target LOO elpd diff for every pair in one named group."""
    output_dir = Path(results_pth) / "hddm_compare"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{group_name}] Running LOO comparison for {len(pairs)} pair(s)...", flush=True)
    compare_df = pd.DataFrame(
        compare_pair(reference_model, target_model, dataset, output_dir)
        for reference_model, target_model in pairs
    )
    compare_df.to_csv(
        summary_dir / f"{dataset}_{group_name}_model_comparison_loo.csv", index=False
    )
    print(compare_df.to_string(index=False), flush=True)


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

    for group_name, pairs in MODEL_GROUPS.items():
        run_group_comparison(
            group_name=group_name,
            pairs=pairs,
            dataset=args.dataset_name,
            results_pth=args.results_pth,
        )


if __name__ == "__main__":
    main()
