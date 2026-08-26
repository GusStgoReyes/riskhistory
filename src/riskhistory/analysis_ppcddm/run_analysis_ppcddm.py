import argparse
import pandas as pd
from filelock import FileLock
from pathlib import Path

from riskhistory.analysis_ppcddm.model_factory import get_model
from riskhistory.config import behavioral_data_pth, preprocessed_data_pth, results_pth

# ---------------------------------------------------------------------------
# Models to simulate
# ---------------------------------------------------------------------------
MODELS = {
    "Baseline":                    "HDDM_Base",
    "Starting Point Bias":         "HDDM_Base_Lag1-stp",
    "Drift Bias":                  "HDDM_Base_Lag1-drift",
    "Starting Point + Drift Bias": "HDDM_Base_Lag1-stpdrift",
    "Drift Bias (pupil-clean population)":         "HDDM_Base_Lag1-drift_Pupil",
    "Pupil x Fixed Drift (drift-only)":            "HDDM_Base_Lag1-drift_Pupil_drift",
    "Starting Point Bias (pupil-clean population)": "HDDM_Base_Lag1-stp_Pupil",
    "Pupil x Starting Point (stp-only)":           "HDDM_Base_Lag1-stp_Pupil_stp",
}

# Models whose model_id contains this marker need the pupil-merged
# behavioral data
PUPIL_MODEL_MARKER = "_Pupil"

DATASET_NAMES = ["equalIndifference", "equalRange", "sheng2020"]
NUM_SAMPLES_PER_CONDITION = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_job_manifest(behavioral_data):
    """Sorted list of (dataset, sub) pairs — index = SLURM_ARRAY_TASK_ID."""
    jobs = []
    for ds in DATASET_NAMES:
        subs = sorted(behavioral_data[behavioral_data["dataset"] == ds]["sub"].unique())
        jobs.extend((ds, int(s)) for s in subs)
    return jobs


def load_behavioral_data_pupilclean():
    """Behavioral data with the pupil GLM amplitude merged in as a `pupil` column
    """
    behavioral_data = (
        pd.read_csv(behavioral_data_pth)
        [["sub", "dataset", "trial", "run", "gain_centered", "loss_centered",
          "choice_prev1", "accept", "repeat", "rt", "exclude"]]
        .query("exclude == False")
    )
    # Rename gain_centered/loss_centered (no _centered)
    behavioral_data["gain"] = behavioral_data["gain_centered"]
    behavioral_data["loss"] = behavioral_data["loss_centered"]

    pupil_betas = (
        pd.read_csv(preprocessed_data_pth / "pupilbetas_glm_preprocessed.csv")
        [["sub", "trial", "run", "dataset", "pupil", "exclude_startAligned"]]
        .query("exclude_startAligned == False")
        .drop(columns="exclude_startAligned")
    )

    return behavioral_data.merge(pupil_betas, on=["sub", "trial", "run", "dataset"], how="inner")


def setup_model_parameters(model_name, sub_params, group_params, dataset, sub):
    model = get_model(model_name)
    sub_params  = sub_params.query(f"sub == {sub} & dataset == '{dataset}'")[["param_name", "param_value"]]
    group_params = group_params.query(f"dataset == '{dataset}'")[["param_name", "param_value"]]
    group_params = group_params[~group_params["param_name"].str.contains("_std")]

    for df in (sub_params, group_params):
        df["param_name"] = (df["param_name"]
                            .str.replace(":", "_")
                            .str.replace("_centered", "")
                            .str.replace(r"\bt\b", "ter", regex=True)
                            .str.replace(r"\bz_Intercept\b", "z", regex=True))

    combined = (pd.concat([group_params, sub_params])
                .drop_duplicates(subset=["param_name"], keep="last"))
    
    param_dict = dict(zip(combined["param_name"], combined["param_value"]))

    # PyDDM Boundary is half of HDDM's a -- for the pupil-arousal _boundary
    # mechanism, a_Intercept/a_pupil replace "a" on the same scale, so both
    # need the same halving (linear in the boundary function, so halving
    # each term is equivalent to halving the whole a_Intercept + a_pupil *
    # pupil expression).
    for key in ("a", "a_Intercept", "a_pupil"):
        if key in param_dict:
            param_dict[key] = param_dict[key] / 2.0
        
    model.set_parameters(param_dict)
    return model


def generate_simulated_data(model, dataset, sub, behavioral_data,
                             num_samples_per_condition=NUM_SAMPLES_PER_CONDITION):
    sub_data   = behavioral_data.query(f"sub == {sub} & dataset == '{dataset}'")
    # "pupil" is already carried through automatically for models that use
    # it as a genuine DDM condition (self.conditions); this covers the case
    # where it isn't one (e.g. HDDM_Base_Lag1-stpdrift_Pupil, which has no
    # pupil interaction term but is still simulated on the pupil-clean
    # population) -- extra_columns is a no-op there since generate_data()
    # skips names already covered by self.conditions.
    extra_columns = ["pupil"] if "pupil" in behavioral_data.columns else []
    trials_df  = model.generate_data(sub_data, samples_per_condition=num_samples_per_condition,
                                     extra_columns=extra_columns)
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

def run_subject(dataset_name: str, sub: int, behavioral_data: pd.DataFrame,
                behavioral_data_pupilclean: pd.DataFrame,
                models: dict = MODELS) -> None:
    """Run each model in `models` for one (dataset, sub) pair and append
    results to shared CSVs. Pupil-arousal models (model_id containing
    PUPIL_MODEL_MARKER) use behavioral_data_pupilclean instead of
    behavioral_data, since they need a "pupil" condition column and a
    trial population restricted to pupil-clean trials."""
    dataset_behavioral_data = behavioral_data.query(f"dataset == '{dataset_name}' & sub == {sub}")
    dataset_behavioral_data_pupilclean = behavioral_data_pupilclean.query(
        f"dataset == '{dataset_name}' & sub == {sub}"
    )

    for model_name, model_id in models.items():
        is_pupil_model = PUPIL_MODEL_MARKER in model_id
        sub_data = dataset_behavioral_data_pupilclean if is_pupil_model else dataset_behavioral_data

        if sub_data.empty:
            population = "pupil-clean" if is_pupil_model else "behavioral"
            print(f"  [{dataset_name} / sub-{sub}] model: {model_name} -- "
                 f"no {population} trials for this subject, skipping")
            continue

        print(f"  [{dataset_name} / sub-{sub}] model: {model_name}")

        sub_params_pth   = results_pth / "analysis_ddm" / "hddm_params" / f"{model_id}_{dataset_name}_subj_params.csv"
        group_params_pth = results_pth / "analysis_ddm" / "hddm_stats"  / f"{model_id}_{dataset_name}_model_stats.csv"

        sub_params   = pd.read_csv(sub_params_pth)
        group_params = pd.read_csv(group_params_pth)

        model_instance = setup_model_parameters(model_id, sub_params, group_params,
                                                dataset_name, sub)
        trials_df = generate_simulated_data(model_instance, dataset_name, sub, sub_data)

        out_path = results_pth / "analysis_ppcddm" / f"{model_id}_simulated_data.csv"
        _append_with_lock(out_path, trials_df)
        print(f"    Saved {len(trials_df)} rows -> {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPC-DDM simulation")
    parser.add_argument("--job_index", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID: index into (dataset, sub) manifest")
    parser.add_argument("--print_njobs", action="store_true",
                        help="Print total number of jobs and exit (for sbatch array sizing)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model_ids to simulate, e.g. 'HDDM_Base' "
                             "or 'HDDM_Base,HDDM_Base_Lag1-full' (default: all in MODELS)")
    args = parser.parse_args()

    if args.models:
        selected_ids = [m.strip() for m in args.models.split(",")]
        models = {name: model_id for name, model_id in MODELS.items()
                 if model_id in selected_ids}
        unknown = set(selected_ids) - set(models.values())
        if unknown:
            raise ValueError(f"Unknown model_id(s) in --models: {sorted(unknown)}. "
                             f"Valid options: {sorted(MODELS.values())}")
    else:
        models = MODELS

    behavioral_data = (
        pd.read_csv(behavioral_data_pth)
        [["sub", "dataset", "trial", "gain_centered", "loss_centered", "choice_prev1", "accept", "repeat", "rt", "exclude"]]
        .query("exclude == False")
    )
    # Rename gain_centered/loss_centered (no _centered)
    behavioral_data["gain"] = behavioral_data["gain_centered"]
    behavioral_data["loss"] = behavioral_data["loss_centered"]

    behavioral_data_pupilclean = load_behavioral_data_pupilclean()

    # Union of both populations' (dataset, sub) pairs -- pupil-clean is
    # expected to be a subset of the plain population (extra pupil
    # exclusion on top of the same behavioral exclusion), but the union is
    # taken defensively rather than assumed.
    jobs = sorted(set(_build_job_manifest(behavioral_data))
                 | set(_build_job_manifest(behavioral_data_pupilclean)))

    if args.print_njobs:
        print(len(jobs))
        raise SystemExit(0)

    if args.job_index is not None:
        if args.job_index >= len(jobs):
            print(f"job_index {args.job_index} >= n_jobs {len(jobs)}, nothing to do.")
            raise SystemExit(0)
        jobs = [jobs[args.job_index]]

    for dataset_name, sub in jobs:
        print(f"\nProcessing dataset={dataset_name}, sub={sub}")
        run_subject(dataset_name, sub, behavioral_data, behavioral_data_pupilclean, models)

    print("\nDone.")
