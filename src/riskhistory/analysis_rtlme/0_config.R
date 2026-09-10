# =============================================================================
# Choice History Under Risk: Config for RT LME Analysis
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_rtlme"
analysis_data_path    <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

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

# =============================================================================
# Source shared settings and utility functions
# =============================================================================
shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))

# add_transition_regressor() -- same current x previous choice transition
# factor used in analysis_pupiltimeseries / analysis_pupilbetas
source(file.path(shared_utils_path, "utils_createtransitionregressor.R"))

# =============================================================================
# Load and prepare data
# =============================================================================
message("Loading behavioral data...")
behavioral_data_full <- load_behavioral_data()

# add_subjectivevalues() merges the M1 baseline betas, computes sv, and
# subsets to non-excluded trials
message("Computing subjective values...")
pth_betas <- file.path(results_path, "analysis_acceptglme",
                       "1_baselinebehavior_betas.csv")
rtlme_data <- add_subjectivevalues_and_regressors(df = behavioral_data_full, pth_betas = pth_betas)
rtlme_data <- subset(rtlme_data, !exclude)

# 4-level current x previous choice transition factor
message("Creating transition (current x previous choice) regressor...")
rtlme_data <- add_transition_regressor(rtlme_data)

message("Config loaded successfully.")
for (ds in datasets) {
  n_subs <- length(unique(rtlme_data$sub[rtlme_data$dataset == ds]))
  n_trials <- sum(rtlme_data$dataset == ds)
  message(sprintf("Dataset: %s, Unique Participants: %d, Trials: %d",
                  ds, n_subs, n_trials))
}
