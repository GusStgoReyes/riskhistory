import pandas as pd
from riskhistory.setup_pupilbetas.utils_models import highpass_filter
from riskhistory.setup_pupilbetas.utils_get_data import (get_all_participants,
                                                            get_all_timings_df,
                                                            get_participant_pupil_data)
from riskhistory.setup_pupilbetas.utils_glm import glm_single_sub
from riskhistory.setup_pupilbetas.utils_prf import get_prf
from riskhistory.setup_pupilbetas.utils_savedata import save_glm_to_csv
from riskhistory.config import results_pth

# Parameters for highpass filtering of pupil data
CUTOFF = 0.05
FS = 25
ORDER = 3

# Parameters for PRF convolution
PRF_MAX_SECS = 4
PRF_WIDTH = 10.1
PRF_TIME_TO_PEAK = 930

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

    start_timings_df = start_timings_df.merge(participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner")
    response_timings_df = response_timings_df.merge(participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner")

    chosen_prf = get_prf(max_secs=PRF_MAX_SECS, width=PRF_WIDTH, time_to_peak=PRF_TIME_TO_PEAK)

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
            pupil_data["pupil"] = highpass_filter(pupil_data["pupil_filtered"].values,
                                                  cutoff=CUTOFF, fs=FS, order=ORDER)
            all_pupil_data[run] = pupil_data

        glm_results_sub, vif_results_sub = glm_single_sub(
            all_pupil_data, chosen_prf,
            start_timings, response_timings,
            sub, runs, dataset
        )
        glm_results.append(glm_results_sub)
        vif_results.append(vif_results_sub)

    glm_results_df = pd.concat(glm_results, ignore_index=True)
    vif_df = pd.concat(vif_results, ignore_index=True)

    # Summary: mean and SD of VIF across all runs, by dataset
    vif_summary = (
        vif_df.groupby("dataset")[["mean_vif_decision", "mean_vif_response"]]
        .agg(["mean", "mean"])
        .round(3)
    )
    print("\nVIF summary (mean_vif across runs):")
    print(vif_summary.to_string())

    # Load exclusions and behavioral data for saving
    exclusions = pd.read_csv(results_pth / "setup_pupildata" / "pupil_data_statistics_with_exclusions.csv")

    # Save GLM results
    save_glm_to_csv(glm_results_df, exclusions)

    # Save VIF results
    vif_out = results_pth / "setup_pupilbetas"
    vif_out.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(vif_out / "design_vif_per_run.csv", index=False)
    vif_summary.to_csv(vif_out / "design_vif_summary.csv")
    print(f"Saved VIF results to {vif_out}")

if __name__ == "__main__":
    main()