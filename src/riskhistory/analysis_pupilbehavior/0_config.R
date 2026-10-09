# =============================================================================
# Config for Pupil-Behavior Analysis
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_pupilbehavior"
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
library(tibble)           # used by compare_models
library(marginaleffects)
library(bayestestR)       # used by compare_models

# =============================================================================
# Source shared settings and utility functions
# =============================================================================
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))
source(file.path(shared_utils_path, "utils_modelcomparison.R"))

# =============================================================================
# Load and prepare behavioral + pupil data
# =============================================================================
message("Loading behavioral and pupil betas data...")
data_pupilbehavior <- load_behavioral_pupil_data()

message(sprintf("Pupil-behavior data: %d rows", nrow(data_pupilbehavior)))
for (ds in datasets) {
  ds_data <- data_pupilbehavior[data_pupilbehavior$dataset == ds, ]
  message(sprintf("  %s: %d subjects, %d trials",
                  ds, length(unique(ds_data$sub)), nrow(ds_data)))
}

message("Config loaded successfully.")
