# =============================================================================
# Choice History Under Risk: Config for Pupil LME Analysis
# =============================================================================
# This file contains analysis-specific settings for the pupil LME
# analysis. Paths are imported from the base config.
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupillme"
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
source(file.path(utils_path, "utils_addsubjectivevalues.R"))
source(file.path(utils_path, "utils_scalestimuli.R"))
source(file.path(utils_path, "utils_addentropy.R"))
source(file.path(utils_path, "utils_modelanalytics.R"))
source(file.path(utils_path, "utils_createpupilregressors.R"))
source(file.path(utils_path, "utils_timepointmodels.R"))


# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

# Define datasets to analyze
datasets <- c("equalIndifference", "equalRange", "sheng2020")

# Multiple comparisons method
p_method <- "BH"  # Benjamini-Hochberg

# =============================================================================
# Load and prepare behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))

# Scale stimuli (creates gain_centered, loss_centered)
message("Scaling stimuli...")
behavioral_data <- scale_stimuli(behavioral_data)

# Load betas from baseline behavior model
message("Loading subject-specific betas...")
betas_path <- file.path(results_path, "analysis_acceptglme",
                        "1_baselinebehavior_betas.csv")
betas_sub <- read.csv(betas_path)

# Merge betas into behavioral data
behavioral_data <- merge(behavioral_data, betas_sub,
                         by = c("sub", "dataset"))

# Create pupil regressors (sv, entropy, logRT, switch, choice_prev1)
message("Creating pupil regressors...")
behavioral_data <- create_pupil_regressors(behavioral_data)

# =============================================================================
# Load pupil data
# =============================================================================
message("Loading pupil data...")

pupil_data_start <- read.csv(file.path(analysis_data_path,
                                       "preprocessed",
                                       "pupil_data_alignedStartTime.csv"))

pupil_data_response <- read.csv(file.path(analysis_data_path,
                                          "preprocessed",
                                          "pupil_data_alignedResponseTime.csv"))

# Remove baseline from pupil size
pupil_data_start <- pupil_data_start %>%
  mutate(pupil_size = pupil_size - baseline_pupil)

pupil_data_response <- pupil_data_response %>%
  mutate(pupil_size = pupil_size - baseline_pupil)

# Rename exclude column in pupil data to avoid conflict
pupil_data_start <- pupil_data_start %>%
  rename(pupil_exclude = exclude)

pupil_data_response <- pupil_data_response %>%
  rename(pupil_exclude = exclude)

# =============================================================================
# Merge behavioral and pupil data
# =============================================================================
message("Merging behavioral and pupil data...")

# Select relevant behavioral columns for merge
behavioral_cols <- c("sub", "trial", "dataset", "sv", "entropy", "logRT",
                     "choice_prev1", "switch", "choice", "exclude")
behavioral_subset <- behavioral_data[, behavioral_cols]

# Merge with pupil data
data_start <- merge(pupil_data_start, behavioral_subset,
                    by = c("sub", "trial", "dataset"))

data_response <- merge(pupil_data_response, behavioral_subset,
                       by = c("sub", "trial", "dataset"))

# =============================================================================
# Apply exclusion criteria
# =============================================================================
message("Applying exclusion criteria...")

# Combined exclusion: exclude if either behavioral or pupil exclusion flag is TRUE
data_start <- data_start %>%
  mutate(combined_exclude = (exclude == "True") | (pupil_exclude == TRUE))

data_response <- data_response %>%
  mutate(combined_exclude = (exclude == "True") | (pupil_exclude == TRUE))

# Filter to non-excluded trials
data_start_clean <- data_start %>%
  filter(!combined_exclude)

data_response_clean <- data_response %>%
  filter(!combined_exclude)

message(sprintf("Start-aligned data: %d rows total, %d rows after exclusion",
                nrow(data_start), nrow(data_start_clean)))
message(sprintf("Response-aligned data: %d rows total, %d rows after exclusion",
                nrow(data_response), nrow(data_response_clean)))

# =============================================================================
# Get unique timebins
# =============================================================================
timebins_start <- sort(unique(data_start_clean$timebin))
timebins_response <- sort(unique(data_response_clean$timebin))

message(sprintf("Start-aligned timebins: %d (range: %.2f to %.2f)",
                length(timebins_start), min(timebins_start), max(timebins_start)))
message(sprintf("Response-aligned timebins: %d (range: %.2f to %.2f)",
                length(timebins_response), min(timebins_response), max(timebins_response)))

message("Config loaded successfully.")
