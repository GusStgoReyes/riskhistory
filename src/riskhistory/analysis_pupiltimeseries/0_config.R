# =============================================================================
# Config for Pupil Timeseries Analysis
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_pupiltimeseries"
analysis_results_path <- file.path(results_path, analysis_name)
utils_path            <- dirname(sys.frame(1)$ofile)
shared_utils_path     <- file.path(repo_path, "src", "riskhistory",
                                   "analysis_utils")

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)         # p-values for lmer fixed effects
library(broom.mixed)      # used by fit_timepoint_model
library(dplyr)
library(marginaleffects)

# =============================================================================
# Source shared settings and utility functions
# =============================================================================
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_createtransitionregressor.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))
source(file.path(shared_utils_path, "utils_contrasts.R"))
source(file.path(utils_path, "utils_timepointmodels.R"))

# =============================================================================
# Load behavioral and pupil data
# =============================================================================
message("Loading behavioral data and computing subjective values...")

behavioral_data_pupil <- load_behavioral_pupil_data()

message("Loading pupil data...")
pupil_data_start    <- load_pupil_data("start")
pupil_data_response <- load_pupil_data("response")

merge_pupil_behavior <- function(pupil_data) {
  merge(pupil_data, behavioral_data_pupil,
        by = c("sub", "trial", "dataset")) %>%
    add_transition_regressor()
}

data_start_clean    <- merge_pupil_behavior(pupil_data_start)
data_response_clean <- merge_pupil_behavior(pupil_data_response)

for (ds in datasets) {
  sub_summary <- data_start_clean %>%
    filter(dataset == ds) %>%
    group_by(sub) %>%
    summarise(n_trials    = n_distinct(trial),
              mean_accept = mean(accept, na.rm = TRUE),
              mean_rt     = mean(rt, na.rm = TRUE),
              .groups     = "drop")
  message(sprintf(
    "  %s: %d subjects, %.1f trials/subject, %.2f mean accept, %.3f mean RT",
    ds, nrow(sub_summary), mean(sub_summary$n_trials),
    mean(sub_summary$mean_accept), mean(sub_summary$mean_rt)
  ))
}

# =============================================================================
# Timebins
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
