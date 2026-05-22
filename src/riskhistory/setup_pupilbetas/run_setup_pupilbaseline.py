import numpy as np
import pandas as pd
from riskhistory.setup_pupilbetas.utils_models import lowpass_filter
from riskhistory.setup_pupilbetas.utils_get_data import (get_all_participants,
                                                          get_all_timings_df,
                                                          get_participant_pupil_data)
from riskhistory.config import preprocessed_data_pth, results_pth

LOWPASS_CUTOFF = 0.1
FS = 25
ORDER = 3

BASELINE_START = -0.5
BASELINE_END = 0.0

DATASET_RUNS = {
    "equalIndifference": [1, 2, 3, 4],
    "equalRange": [1, 2, 3, 4],
    "sheng2020": [1, 2],
}


def extract_baseline_run(pupil_data, start_timings_run):
    time_vector = pupil_data["time_secs"].values
    pupil_lp = lowpass_filter(pupil_data["pupil_filtered"].values, cutoff=LOWPASS_CUTOFF, fs=FS, order=ORDER)

    records = []
    for _, trial_row in start_timings_run.iterrows():
        trial = trial_row["trial"]
        onset = trial_row["time_secs"]
        mask = (time_vector >= onset + BASELINE_START) & (time_vector < onset + BASELINE_END)
        if mask.sum() == 0:
            baseline = np.nan
        else:
            baseline = pupil_lp[mask].mean()
        records.append({"trial": trial, "pupil_baseline": baseline})

    return pd.DataFrame(records)


def main():
    participants, exclusions = get_all_participants()
    start_timings_df = get_all_timings_df(name="start")
    start_timings_df = start_timings_df.merge(participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner")

    results = []
    for _, row in participants.iterrows():
        sub = row["sub"]
        dataset = row["dataset"]
        runs = DATASET_RUNS.get(dataset, None)
        print(f"Processing participant {sub} from dataset {dataset}...")

        start_timings = start_timings_df.query("sub == @sub and dataset == @dataset")

        for run in runs:
            pupil_data = get_participant_pupil_data(sub, run, dataset)
            start_timings_run = start_timings.query("run == @run")

            baseline_df = extract_baseline_run(pupil_data, start_timings_run)
            baseline_df["run"] = run
            baseline_df["sub"] = sub
            baseline_df["dataset"] = dataset
            results.append(baseline_df)

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df[["dataset", "sub", "run", "trial", "pupil_baseline"]]
    results_df["pupil_baseline"] = results_df.groupby(["sub", "run"])["pupil_baseline"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    exclusions = exclusions[["dataset", "sub", "run", "trial", "exclude_startAligned"]]
    results_df = results_df.merge(exclusions, on=["dataset", "sub", "run", "trial"], how="left")

    preprocessed_data_pth.mkdir(parents=True, exist_ok=True)
    out_path = preprocessed_data_pth / "pupilbaseline_preprocessed.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved baseline results to {out_path}")


if __name__ == "__main__":
    main()
