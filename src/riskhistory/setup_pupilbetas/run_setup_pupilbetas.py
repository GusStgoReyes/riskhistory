import pandas as pd
from riskhistory.setup_pupilbetas.utils_get_data import (get_all_participants,
                                                            get_all_timings_df,
                                                            get_participant_pupil_data,
                                                            get_behavioral_data)
from riskhistory.setup_pupilbetas.utils_lss import lss_single_sub
from riskhistory.setup_pupilbetas.utils_prf import get_prf
from riskhistory.config import (results_pth,
                                preprocessed_data_pth)


def save_betas_to_csv(results_df, exclusions, behavioral_data):
    # Save all of the extracted betas into the results
    results_dir = results_pth / "setup_pupilbetas"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "pupilbetas_lss_results.csv", index=False)

    # Merge behavioral and pupil exclusion criteria
    exclusions = exclusions[["dataset", "sub", "trial", "run", "exclude_startAligned"]]
    behavioral_data = behavioral_data[["dataset", "sub", "trial", "run", "exclude"]]
    exclude = exclusions.merge(behavioral_data, on=["dataset", "sub", "trial", "run"], how="left")
    exclude["final_exclude"] = exclude["exclude_startAligned"] | exclude["exclude"]

    # Query the trial_betas_names that contains "target"
    target_betas = results_df[results_df['trial_betas_names'].str.contains("target")]
    target_betas.loc[:, "trial"] = target_betas.loc[:, "target_trial"]
    target_betas = target_betas.drop(columns=["target_trial", "trial_betas_names"])
    target_betas = target_betas.reset_index(drop=True)
    target_betas = target_betas.rename(columns={"trial_betas_estimates": "pupil"})

    # zscore within subject
    target_betas["pupil"] = target_betas.groupby("sub")["pupil"].transform(lambda x: (x - x.mean()) / x.std())

    target_betas = target_betas.merge(exclude[["dataset", "sub", "trial", "final_exclude"]], on=["dataset", "sub", "trial"], how="left")

    # Save target betas to preprocessed data folder
    preprocessed_data_pth.mkdir(parents=True, exist_ok=True)
    target_betas.to_csv(preprocessed_data_pth / "pupilbetas_lss_preprocessed.csv", index=False)


def main():
    participants = get_all_participants()
    start_timings_df = get_all_timings_df(name = "start")
    response_timings_df = get_all_timings_df(name = "response")

    # Only select participants in the participants df (with columns sub/dataset)
    start_timings_df = start_timings_df.merge(participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner")
    response_timings_df = response_timings_df.merge(participants[["dataset", "sub"]], on=["dataset", "sub"], how="inner")

    # Get the PRF of interest
    chosen_prf = get_prf(max_secs = 4, width = 10.1, time_to_peak = 930) 

    # Loop through each participant and run LSS
    results = []
    for _, row in participants.iterrows():
        sub = row["sub"]
        dataset = row["dataset"]
        runs = [1, 2, 3, 4] if dataset in ["equalIndifference", "equalRange"] else [1, 2] if dataset == "sheng2020" else None
        print(f"Processing participant {sub} from dataset {dataset}...")

        # Get the timings for this participant
        start_timings = start_timings_df.query("sub == @sub and dataset == @dataset")
        response_timings = response_timings_df.query("sub == @sub and dataset == @dataset")
        
        # Gather all the pupil data for this participant across runs into a dictionary
        all_pupil_data = {}
        for run in runs: 
            pupil_data = get_participant_pupil_data(sub, run, dataset)
            all_pupil_data[run] = pupil_data
        
        # Run LSS for this participant
        results_sub = lss_single_sub(all_pupil_data,
                                    chosen_prf,
                                    start_timings, 
                                    response_timings,
                                    sub,
                                    runs,
                                    dataset)
        
        results.append(results_sub)
    
    # Combine results across participants into a single dataframe
    results_df = pd.concat(results, ignore_index=True)

    # Load exclusions and behavioral data for saving
    exclusions = pd.read_csv(results_pth / "setup_pupildata" / "pupil_data_statistics_with_exclusions.csv")
    behavioral_data = get_behavioral_data()

    save_betas_to_csv(results_df, exclusions, behavioral_data)


if __name__ == "__main__":
    main()