import pandas as pd
from riskhistory.setup_pupilbetas.utils_models import highpass_filter
from riskhistory.setup_pupilbetas.utils_get_data import (
    get_all_participants,
    get_all_timings_df,
    get_participant_pupil_data,
)
from riskhistory.setup_pupilbetas.utils_glm_basis import glm_single_sub_basis
from riskhistory.setup_pupilbetas.utils_prf_basis import get_prf_basis
from riskhistory.setup_pupilbetas.utils_savedata_basis import save_glm_basis_to_csv
from riskhistory.config import results_pth

# Parameters for highpass filtering of pupil data
CUTOFF = 0.05
FS = 25
ORDER = 3

# Parameters for PRF basis generation
PRF_MAX_SECS = 4
PRF_N_COMPONENTS = 3
PRF_WIDTH_RANGE = (5, 15)
PRF_TIME_TO_PEAK_RANGE = (500, 1300)
PRF_N_GRID = 20

# Runs per dataset
DATASET_RUNS = {
    "equalIndifference": [1, 2, 3, 4],
    "equalRange": [1, 2, 3, 4],
    "sheng2020": [1, 2],
}


def main():
    participants = get_all_participants()
    start_timings_df = get_all_timings_df(name="start")
    response_timings_df = get_all_timings_df(name="response")

    start_timings_df = start_timings_df.merge(
        participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner"
    )
    response_timings_df = response_timings_df.merge(
        participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner"
    )

    basis, explained_variance = get_prf_basis(
        max_secs=PRF_MAX_SECS,
        n_components=PRF_N_COMPONENTS,
        width_range=PRF_WIDTH_RANGE,
        time_to_peak_range=PRF_TIME_TO_PEAK_RANGE,
        n_grid=PRF_N_GRID,
    )
    print(f"PRF basis — explained variance per component: {explained_variance.round(3)}")

    glm_results = []
    vif_results = []
    for _, row in participants.iterrows():
        sub = row["sub"]
        dataset = row["dataset"]
        runs = DATASET_RUNS.get(dataset, None)
        print(f"Processing participant {sub} from dataset {dataset}...")

        start_timings = start_timings_df.query("sub == @sub and dataset == @dataset")
        response_timings = response_timings_df.query("sub == @sub and dataset == @dataset")

        all_pupil_data = {}
        for run in runs:
            pupil_data = get_participant_pupil_data(sub, run, dataset)
            pupil_data["pupil"] = highpass_filter(
                pupil_data["pupil_filtered"].values, cutoff=CUTOFF, fs=FS, order=ORDER
            )
            all_pupil_data[run] = pupil_data

        glm_results_sub, vif_results_sub = glm_single_sub_basis(
            all_pupil_data, basis,
            start_timings, response_timings,
            sub, runs, dataset,
        )
        glm_results.append(glm_results_sub)
        vif_results.append(vif_results_sub)

    glm_results_df = pd.concat(glm_results, ignore_index=True)
    vif_df = pd.concat(vif_results, ignore_index=True)

    vif_summary = (
        vif_df.groupby("dataset")[["mean_vif_decision", "mean_vif_response"]]
        .agg("mean")
        .round(3)
    )
    print("\nVIF summary (mean_vif across runs):")
    print(vif_summary.to_string())

    exclusions = pd.read_csv(
        results_pth / "setup_pupildata" / "pupil_data_statistics_with_exclusions.csv"
    )

    save_glm_basis_to_csv(glm_results_df, exclusions)

    vif_out = results_pth / "setup_pupilbetas"
    vif_out.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(vif_out / "design_vif_per_run_basis.csv", index=False)
    vif_summary.to_csv(vif_out / "design_vif_summary_basis.csv")
    print(f"Saved VIF results to {vif_out}")


if __name__ == "__main__":
    main()
