from riskhistory.config import results_pth, derivatives_pth, behavioral_data_pth
import pandas as pd

def get_all_participants():
    exclusion_pth = results_pth / "setup_pupildata" / "pupil_data_statistics_with_exclusions.csv"
    exclusions = pd.read_csv(exclusion_pth)
    datasets = ["equalIndifference", "equalRange", "sheng2020"]
    rows = []
    for dataset in datasets:
        valid = (exclusions
                 .query(f"dataset == '{dataset}' and exclude_startAligned == False")["sub"]
                 .unique()
                 .tolist())
        print(f"Num of valid participants for {dataset}: {len(valid)}")
        rows.extend({"dataset": dataset, "sub": sub} for sub in valid)
    return pd.DataFrame(rows), exclusions

def dataset_subs_timings(timings_df, dataset):
    # NARPS contains both conditions in one file; split by subject parity
    if dataset == "equalIndifference":
        timings_df = timings_df.query("sub_id % 2 == 1")
    elif dataset == "equalRange":
        timings_df = timings_df.query("sub_id % 2 == 0")

    # Trial numbers in the timing files are run-local; convert to global trial index
    num_trials = 64 if dataset in ["equalIndifference", "equalRange"] else 100 if dataset == "sheng2020" else None
    timings_df = timings_df.copy()
    timings_df["trial"] = timings_df["trial"] + (timings_df["run_id"] - 1) * num_trials
    timings_df = timings_df.rename(columns={"sub_id": "sub", "run_id": "run"})

    return timings_df

def get_start_timings_df(dataset):
    if dataset in ["equalIndifference", "equalRange"]:
        timings_pth = derivatives_pth / "pupil" / "narps" / "timeseries" / "narps_start_timings.csv"
    elif dataset == "sheng2020":
        timings_pth = derivatives_pth / "pupil" / "sheng2020" / "timeseries" / "sheng2020_start_timings.csv"
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Please choose from 'equalIndifference', 'equalRange', or 'sheng2020'.")
    
    timings_df = pd.read_csv(timings_pth)
    timings_df = dataset_subs_timings(timings_df, dataset)
    timings_df["dataset"] = dataset
    
    return timings_df

def get_response_timings_df(dataset):
    if dataset in ["equalIndifference", "equalRange"]:
        timings_pth = derivatives_pth / "pupil" / "narps" / "timeseries" / "narps_response_timings.csv"
    elif dataset == "sheng2020":
        timings_pth = derivatives_pth / "pupil" / "sheng2020" / "timeseries" / "sheng2020_response_timings.csv"
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Please choose from 'equalIndifference', 'equalRange', or 'sheng2020'.")
    
    timings_df = pd.read_csv(timings_pth)
    timings_df = dataset_subs_timings(timings_df, dataset)
    timings_df["dataset"] = dataset
    
    return timings_df

def get_all_timings_df(name = "start", min_rt = 0.2):
    datasets = ["equalIndifference", "equalRange", "sheng2020"]
    all_timings_df = pd.DataFrame()
    for dataset in datasets:
        if name == "start":
            timings_df = get_start_timings_df(dataset)
        elif name == "response":
            timings_df = get_response_timings_df(dataset)
        else:
            raise ValueError("Name must be either 'start' or 'response'.")
        
        all_timings_df = pd.concat([all_timings_df, timings_df], ignore_index=True)
    
    # Add duration (rt) column to the timings
    behavioral_data = get_behavioral_data()
    all_timings_df = all_timings_df.merge(behavioral_data[["dataset", "sub", "run", "trial", "rt"]], on=["dataset", "sub", "run", "trial"], how="left")
    all_timings_df = all_timings_df.rename(columns={"rt": "duration"}, inplace=False)
    # If duration is NaN or <0.2, set to 0
    all_timings_df["duration"] = all_timings_df["duration"].fillna(0)
    all_timings_df.loc[all_timings_df["duration"] < min_rt, "duration"] = 0
    
    return all_timings_df

def get_behavioral_data():
    behavioral_df = pd.read_csv(behavioral_data_pth)
    return behavioral_df

def get_participant_pupil_data(sub, run, dataset):
    if dataset in ["equalIndifference", "equalRange"]:
        pth = derivatives_pth / "pupil" / "narps" / "timeseries" / f"sub-{sub:03d}_run-{run}_timeseries.csv"
    elif dataset == "sheng2020":
        pth = derivatives_pth / "pupil" / "sheng2020" / "timeseries" / f"sub-{sub:03d}_run-{run}_timeseries.csv"
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Please choose from 'equalIndifference', 'equalRange', or 'sheng2020'.")
    
    pupil_data = pd.read_csv(pth)
    return pupil_data