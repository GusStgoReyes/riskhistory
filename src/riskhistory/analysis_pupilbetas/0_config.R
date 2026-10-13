# =============================================================================
# Config for Pupil Betas Analysis
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_pupilbetas"
analysis_results_path <- file.path(results_path, analysis_name)
shared_utils_path     <- file.path(repo_path, "src", "riskhistory",
                                   "analysis_utils")

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)
library(broom.mixed)      # used by extract_fixed_effects
library(dplyr)
library(marginaleffects)

# =============================================================================
# Source shared settings and utility functions
# =============================================================================
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_createtransitionregressor.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))

# =============================================================================
# Load behavioral + pupil betas data and add the transition regressor
# =============================================================================
message("Loading behavioral and pupil betas data...")
data_betas_clean <- load_behavioral_pupil_data() %>%
  add_transition_regressor()

message(sprintf("Pupil betas data: %d rows", nrow(data_betas_clean)))
for (ds in datasets) {
  sub_summary <- data_betas_clean %>%
    filter(dataset == ds) %>%
    group_by(sub) %>%
    summarise(n_trials    = n(),
              mean_accept = mean(accept, na.rm = TRUE),
              mean_rt     = mean(rt, na.rm = TRUE),
              .groups     = "drop")
  message(sprintf(
    "  %s: %d subjects, %.1f trials/subject, %.2f mean accept, %.3f mean RT",
    ds, nrow(sub_summary), mean(sub_summary$n_trials),
    mean(sub_summary$mean_accept), mean(sub_summary$mean_rt)
  ))
}

message("Config loaded successfully.")
