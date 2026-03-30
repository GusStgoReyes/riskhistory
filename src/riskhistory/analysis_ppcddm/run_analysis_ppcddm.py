import argparse
import pandas as pd
from filelock import FileLock
from pathlib import Path

from riskhistory.analysis_ppcddm.model_factory import get_model
from riskhistory.config import behavioral_data_pth, results_pth

# ---------------------------------------------------------------------------
# Models to simulate
# ---------------------------------------------------------------------------
MODELS = {
    "Baseline":            "HDDM_Collapse",
    "Starting Point Bias": "HDDM_Collapse_Lag1-x0",
    "Drift Bias":          "HDDM_Collapse_Lag1-c0",
    "Stimulus Bias":       "HDDM_Collapse_Lag1-stim",
}
DATASET_NAMES = ["equalIndifference", "equalRange", "sheng2020"]
NUM_SAMPLES_PER_CONDITION = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_job_manifest(behavioral_data: pd.DataFrame) -> list[tuple[str, int]]:
    """Sorted list of (dataset, sub) pairs — index = SLURM_ARRAY_TASK_ID."""
    jobs = []
    for ds in DATASET_NAMES:
        subs = sorted(behavioral_data[behavioral_data["dataset"] == ds]["sub"].unique())
        jobs.extend((ds, int(s)) for s in subs)
    return jobs


def setup_model_parameters(model_name, sub_params, group_params, dataset, sub):
    model = get_model(model_name)
    sub_params  = sub_params.query(f"sub == {sub} & dataset == '{dataset}'")[["param_name", "param_value"]]
    group_params = group_params.query(f"dataset == '{dataset}'")[["param_name", "param_value"]]
    group_params = group_params[~group_params["param_name"].str.contains("_std")]

    for df in (sub_params, group_params):
        df["param_name"] = (df["param_name"]
                            .str.replace(":", "_")
                            .str.replace(r"\bt\b", "ter", regex=True)
                            .str.replace(r"\bz_Intercept\b", "z", regex=True))

    combined = (pd.concat([group_params, sub_params])
                .drop_duplicates(subset=["param_name"], keep="last"))
    model.set_parameters(dict(zip(combined["param_name"], combined["param_value"])))
    return model


def generate_simulated_data(model, dataset, sub, behavioral_data,
                             num_samples_per_condition=NUM_SAMPLES_PER_CONDITION):
    sub_data   = behavioral_data.query(f"sub == {sub} & dataset == '{dataset}'")
    trials_df  = model.generate_data(sub_data, samples_per_condition=num_samples_per_condition)
    trials_df["sub"]     = sub
    trials_df["dataset"] = dataset
    return trials_df


def _append_with_lock(out_path: Path, new_rows: pd.DataFrame) -> None:
    """Append new_rows to out_path CSV using a file lock for safe concurrent writes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(out_path) + ".lock"
    with FileLock(lock_path, timeout=120):
        if out_path.exists():
            existing = pd.read_csv(out_path)
            pd.concat([existing, new_rows], ignore_index=True).to_csv(out_path, index=False)
        else:
            new_rows.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Single-subject entry point (called by each array job)
# ---------------------------------------------------------------------------

def run_subject(dataset_name: str, sub: int, behavioral_data: pd.DataFrame) -> None:
    """Run all models for one (dataset, sub) pair and append results to shared CSVs."""
    dataset_behavioral_data = behavioral_data.query(f"dataset == '{dataset_name}'")

    for model_name, model_id in MODELS.items():
        print(f"  [{dataset_name} / sub-{sub}] model: {model_name}")

        sub_params_pth   = results_pth / "analysis_ddm" / "hddm_params" / f"{model_id}_{dataset_name}_subj_params.csv"
        group_params_pth = results_pth / "analysis_ddm" / "hddm_stats"  / f"{model_id}_{dataset_name}_model_stats.csv"

        sub_params   = pd.read_csv(sub_params_pth)
        group_params = pd.read_csv(group_params_pth)

        model_instance = setup_model_parameters(model_name, sub_params, group_params,
                                                dataset_name, sub)
        trials_df = generate_simulated_data(model_instance, dataset_name, sub,
                                            dataset_behavioral_data)

        out_path = results_pth / "analysis_ppcddm" / f"{model_id}_simulated_data.csv"
        _append_with_lock(out_path, trials_df)
        print(f"    Saved {len(trials_df)} rows → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPC-DDM simulation")
    parser.add_argument("--job_index", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID: index into (dataset, sub) manifest")
    parser.add_argument("--print_njobs", action="store_true",
                        help="Print total number of jobs and exit (for sbatch array sizing)")
    args = parser.parse_args()

    behavioral_data = (
        pd.read_csv(behavioral_data_pth)
        [["sub", "dataset", "gain", "loss", "choice_prev1", "accept", "repeat", "rt", "exclude"]]
        .query("exclude == False")
    )

    jobs = _build_job_manifest(behavioral_data)
    
    if args.job_index is not None:
        if args.job_index >= len(jobs):
            print(f"job_index {args.job_index} >= n_jobs {len(jobs)}, nothing to do.")
            raise SystemExit(0)
        jobs = [jobs[args.job_index]]

    for dataset_name, sub in jobs:
        print(f"\nProcessing dataset={dataset_name}, sub={sub}")
        run_subject(dataset_name, sub, behavioral_data)

    print("\nDone.")
