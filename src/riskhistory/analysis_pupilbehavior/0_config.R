# =============================================================================
# Config for Pupil-Behavior Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupilbehavior"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)
library(broom.mixed)
library(dplyr)
library(tibble)
library(marginaleffects)
library(bayestestR)

# Source shared settings and utility functions
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))
source(file.path(shared_utils_path, "utils_modelcomparison.R"))
source(file.path(shared_utils_path, "utils_createtransitionregressor.R"))
source(file.path(shared_utils_path, "utils_contrasts.R"))

# =============================================================================
# Load and prepare behavioral + pupil data
# =============================================================================
# load_behavioral_pupil_data() merges the M1 baseline betas, computes sv, and
# merges in the pupil GLM betas, filtered to behavioral- and pupil-clean trials.
message("Loading behavioral and pupil betas data...")
data_pupilbehavior <- load_behavioral_pupil_data()

data_pupilbehavior$logRT  <- log(data_pupilbehavior$rt)
data_pupilbehavior$sv_abs <- abs(data_pupilbehavior$sv)

message("Adding choice-history transition regressor...")
data_pupilbehavior <- add_transition_regressor(data_pupilbehavior)

# log(rt), z-scored within sub/dataset (not just sub -- subject IDs repeat
# across datasets, so grouping by sub alone would pool RTs across different
# actual participants).
message("Adding logRT_z regressor (log(rt), z-scored within sub/dataset)...")
data_pupilbehavior <- data_pupilbehavior %>%
  group_by(sub, dataset) %>%
  mutate(logRT_z = scale(log(rt), center = TRUE, scale = TRUE)[, 1]) %>%
  ungroup()

message(sprintf("Pupil-behavior data: %d rows after behavioral + pupil exclusion",
                nrow(data_pupilbehavior)))
for (ds in datasets) {
  ds_data <- data_pupilbehavior %>% filter(dataset == ds)
  n_subs   <- length(unique(ds_data$sub))
  n_trials <- nrow(ds_data)
  message(sprintf("  %s: %d subjects, %d trials", ds, n_subs, n_trials))
}

message("Config loaded successfully.")
