# =============================================================================
# Base Config File for Analyses in R
# =============================================================================

repo_path <- "/Users/gustxsr/Documents/Stanford/PoldrackLab/github/riskhistory"

# Base directory paths
raw_data_path <- file.path(repo_path, "data", "raw")
preprocessed_data_path <- file.path(repo_path, "data", "preprocessed")
derivatives_path <- file.path(repo_path, "data", "derivatives")
results_path <- file.path(repo_path, "results")

# Raw data paths
narps_behavioral_data_path <- file.path(raw_data_path,
                                        "narps",
                                        "behavioral",
                                        "behav_data.csv")
narps_pupil_data_home <- file.path(raw_data_path,
                                   "narps",
                                   "pupil",
                                   "asc")
sheng2020_behavioral_data_path <- file.path(raw_data_path,
                                            "sheng2020",
                                            "behavioral",
                                            "decision.csv")
sheng2020_pupil_data_path <- file.path(raw_data_path,
                                       "sheng2020",
                                       "pupil",
                                       "eye.csv")

# Preprocessed/output paths
behavioral_data_path <- file.path(preprocessed_data_path,
                                  "behavioral_data.csv")
subject_statistics_path <- file.path(preprocessed_data_path,
                                     "subject_statistics.csv")