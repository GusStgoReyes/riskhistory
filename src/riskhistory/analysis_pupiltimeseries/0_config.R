# =============================================================================
# Config for Pupil Timeseries Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupiltimeseries"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

utils_path        <- dirname(sys.frame(1)$ofile)
shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(broom.mixed)
library(dplyr)
library(tibble)
library(lmerTest)
library(marginaleffects)

# Source shared settings and utility functions
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_modelcomparison.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))
source(file.path(shared_utils_path, "utils_createtransitionregressor.R"))
source(file.path(shared_utils_path, "utils_contrasts.R"))

# =============================================================================
# Load and prepare behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- load_behavioral_data()

# Merge betas, compute sv/entropy, and apply behavioral exclusion
message("Loading subject-specific betas and computing subjective value...")
betas_path <- file.path(results_path, "analysis_acceptglme",
                        "1_baselinebehavior_betas.csv")
behavioral_data <- add_subjectivevalues_and_regressors(behavioral_data, betas_path)

# =============================================================================
# Load pupil data
# =============================================================================
message("Loading pupil data...")

pupil_data_start    <- load_pupil_data("start")
pupil_data_response <- load_pupil_data("response")

# =============================================================================
# Merge and apply exclusion criteria
# =============================================================================
message("Merging behavioral and pupil data...")

data_start <- merge(pupil_data_start, behavioral_data,
                    by = c("sub", "trial", "dataset"))

data_response <- merge(pupil_data_response, behavioral_data,
                       by = c("sub", "trial", "dataset"))

message("Applying pupil exclusion criteria...")
data_start_clean <- data_start %>% filter(!pupil_exclude)
data_response_clean <- data_response %>% filter(!pupil_exclude)

message(sprintf("Start-aligned:    %d rows total, %d rows after exclusion",
                nrow(data_start), nrow(data_start_clean)))
message(sprintf("Response-aligned: %d rows total, %d rows after exclusion",
                nrow(data_response), nrow(data_response_clean)))

for (ds in datasets) {
  ds_data <- data_start_clean %>% filter(dataset == ds)
  sub_summary <- ds_data %>%
    group_by(sub) %>%
    summarise(
      n_trials    = n_distinct(trial),
      mean_accept = mean(accept, na.rm = TRUE),
      mean_rt     = mean(rt,     na.rm = TRUE),
      .groups = "drop"
    )
  n_subs      <- nrow(sub_summary)
  mean_trials <- mean(sub_summary$n_trials)
  mean_accept <- mean(sub_summary$mean_accept)
  mean_rt     <- mean(sub_summary$mean_rt)
  message(sprintf("  %s: %d subjects, %.1f trials/subject, %.2f mean accept, %.3f mean RT",
                  ds, n_subs, mean_trials, mean_accept, mean_rt))
}

# =============================================================================
# Create scaled within-subject regressors
# =============================================================================
message("Creating pupil regressors...")
data_start_clean    <- standardize_pupil_regressors(data_start_clean)
data_response_clean <- standardize_pupil_regressors(data_response_clean)

message("Creating transition (current x previous choice) regressor...")
data_start_clean    <- add_transition_regressor(data_start_clean)
data_response_clean <- add_transition_regressor(data_response_clean)

# =============================================================================
# Get unique timebins
# =============================================================================
timebins_start    <- sort(unique(data_start_clean$timebin))
timebins_response <- sort(unique(data_response_clean$timebin))

message(sprintf("Start-aligned timebins:    %d (%.2f to %.2f s)",
                length(timebins_start),
                min(timebins_start), max(timebins_start)))
message(sprintf("Response-aligned timebins: %d (%.2f to %.2f s)",
                length(timebins_response),
                min(timebins_response), max(timebins_response)))

message("Config loaded successfully.")
