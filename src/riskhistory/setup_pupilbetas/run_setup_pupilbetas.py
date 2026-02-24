import numpy as np
import pandas as pd
from riskhistory.setup_pupilbetas.utils_get_data import (get_all_participants,
                                                            get_all_timings_df,
                                                            get_participant_pupil_data)
from riskhistory.setup_pupilbetas.utils_lss import lss_single_sub
from riskhistory.setup_pupilbetas.utils_prf import get_prf

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
    results_df.to_csv("pupilbetas_lss_results.csv", index=False)
        
        
if __name__ == "__main__":
    main()