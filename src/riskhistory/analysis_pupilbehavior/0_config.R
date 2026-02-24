# =============================================================================
# Choice History Under Risk: Config for Pupil-Behavior GLME Analysis
# =============================================================================
# This file loads pupil and behavioral data, computes per-trial pupil mean,
# merges datasets, applies exclusions, and rescales stimuli.
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupilbehavior"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

# Path to utility functions (in this directory)
utils_path <- dirname(sys.frame(1)$ofile)

# Create results directory if it doesn't exist
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

# Source utility functions
source(file.path(utils_path, "utils_pupilpertrial.R"))
source(file.path(utils_path, "utils_scalestimuli.R"))
source(file.path(utils_path, "utils_savefixedeffects.R"))
source(file.path(utils_path, "utils_processdata.R"))
source(file.path(repo_path, "src", "riskhistory", "setup_behavioraldata", "utils_getchoicehistory.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

# Define datasets to analyze
datasets <- c("equalIndifference", "equalRange", "sheng2020")

# Multiple comparisons method
p_method <- "BH"  # Benjamini-Hochberg

# =============================================================================
# Load and process data
# =============================================================================
pupil_beta_path <- file.path(results_path,
                             "setup_pupilbetas",
                             "pupilbetas_lss_results.csv")
behavioral_data_path <- file.path(analysis_data_path,
                                  "preprocessed",
                                  "behavioral_data.csv")

data_clean <- process_data(pupil_beta_path, behavioral_data_path)

# =============================================================================
# Initialize model storage
# =============================================================================
models_list <- list()
for (ds in datasets) {
  models_list[[ds]] <- list()
}

message("Config loaded successfully.")
message(sprintf("Datasets: %s", paste(datasets, collapse = ", ")))
for (ds in datasets) {
  n_trials <- sum(data_clean$dataset == ds)
  n_subs <- length(unique(data_clean$sub[data_clean$dataset == ds]))
  message(sprintf("  %s: %d trials from %d subjects", ds, n_trials, n_subs))
}
