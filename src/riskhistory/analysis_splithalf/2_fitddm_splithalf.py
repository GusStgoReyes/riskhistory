import argparse

from riskhistory.analysis_ppcddm.model_factory import get_model
from riskhistory.analysis_ppcddm.run_analysis_ppcddm import _append_with_lock
from riskhistory.analysis_splithalf.utils import SPLITHALF_RESULTS_PTH, add_half_id
from riskhistory.data import load_behavioral_data

MODEL_ID = "HDDM_Base_Lag1-stpdrift"
DDM_COLUMNS = ["gain", "loss", "choice_prev1", "accept", "rt"]
OUT_FILE = SPLITHALF_RESULTS_PTH / "2_ddm_stpdrift_splithalf_params.csv"


def load_ddm_data():
    data = add_half_id(load_behavioral_data())
    data["gain"] = data["gain_centered"]
    data["loss"] = data["loss_centered"]
    return data


def build_job_manifest(data):
    return sorted(
        (ds, int(sub), int(half_id))
        for (ds, sub, half_id) in data.groupby(["dataset", "sub", "half_id"]).groups
    )


def fit_subject_half(data, dataset, sub, half_id):
    sub_data = data.query(f"dataset == '{dataset}' & sub == {sub} & half_id == {half_id}")
    model = get_model(MODEL_ID)
    model.set_data(sub_data[DDM_COLUMNS].reset_index(drop=True), sub, dataset)
    model.fit_model()

    results = model.get_model_results()
    results.insert(results.columns.get_loc("dataset") + 1, "half_id", half_id)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-half SP-Drift-CH pyddm fits")
    parser.add_argument("--job_index", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID: index into (dataset, sub, half_id) manifest")
    parser.add_argument("--print_njobs", action="store_true",
                        help="Print total number of jobs and exit (for sbatch array sizing)")
    args = parser.parse_args()

    data = load_ddm_data()
    jobs = build_job_manifest(data)

    if args.print_njobs:
        print(len(jobs))
        raise SystemExit(0)

    if args.job_index is not None:
        if args.job_index >= len(jobs):
            print(f"job_index {args.job_index} >= n_jobs {len(jobs)}, nothing to do.")
            raise SystemExit(0)
        jobs = [jobs[args.job_index]]

    for dataset, sub, half_id in jobs:
        print(f"\nFitting dataset={dataset}, sub={sub}, half_id={half_id}")
        results = fit_subject_half(data, dataset, sub, half_id)
        _append_with_lock(OUT_FILE, results)
        print(f"    Saved {len(results)} rows -> {OUT_FILE.name}")

    print("\nDone.")
