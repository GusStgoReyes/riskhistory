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

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

# Define datasets to analyze
datasets <- c("equalIndifference", "equalRange", "sheng2020")

# Multiple comparisons method
p_method <- "BH"  # Benjamini-Hochberg

# =============================================================================
# Load and compute per-trial pupil data
# =============================================================================
message("Loading pupil betas...")

pupil_beta_path <- file.path(analysis_data_path,
                             "preprocessed",
                             "pupil_data_alignedResponseTime.csv")

pupil_per_trial_df <- compute_pupil_per_trial(pupil_data_path)

message(sprintf("Pupil per-trial data: %d trials", nrow(pupil_per_trial_df)))

# =============================================================================
# Load behavioral data
# =============================================================================
message("Loading behavioral data...")

behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))

message(sprintf("Behavioral data: %d trials", nrow(behavioral_data)))

# =============================================================================
# Merge pupil and behavioral data
# =============================================================================
message("Merging pupil and behavioral data...")

merged_data <- merge(behavioral_data, pupil_per_trial_df,
                     by = c("sub", "trial", "dataset"),
                     all.x = FALSE)

message(sprintf("Merged data: %d trials", nrow(merged_data)))

# =============================================================================
# Apply exclusion criteria
# =============================================================================
message("Applying exclusion criteria...")

# Combined exclusion: exclude if either behavioral or pupil exclusion flag
merged_data$combined_exclude <- (merged_data$exclude == "True") |
                                (merged_data$pupil_exclude == TRUE)

# Count exclusions
n_behavioral_exclude <- sum(merged_data$exclude == "True")
n_pupil_exclude <- sum(merged_data$pupil_exclude == TRUE)
n_combined_exclude <- sum(merged_data$combined_exclude)

message(sprintf("Behavioral exclusions: %d", n_behavioral_exclude))
message(sprintf("Pupil exclusions: %d", n_pupil_exclude))
message(sprintf("Combined exclusions: %d", n_combined_exclude))

# Filter to non-excluded trials
data_clean <- merged_data[!merged_data$combined_exclude, ]

message(sprintf("Clean data: %d trials after exclusion", nrow(data_clean)))

# =============================================================================
# Rescale gain and loss
# =============================================================================
message("Rescaling gain and loss...")

data_clean <- scale_stimuli(data_clean)

# Ensure choice_prev1 is numeric
data_clean$choice_prev1 <- as.numeric(data_clean$choice_prev1)

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
