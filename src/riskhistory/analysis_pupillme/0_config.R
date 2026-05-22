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

LMER_CONTROL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load and prepare behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))

# Load betas from baseline behavior model for gain + loss subjective value
message("Loading subject-specific betas...")
betas_path <- file.path(results_path, "analysis_acceptglme",
                        "1_baselinebehavior_betas.csv")
betas_sub <- read.csv(betas_path)

# Merge betas into behavioral data
behavioral_data <- merge(behavioral_data, betas_sub,
                         by = c("sub", "dataset"))

# Add the subjective value to the behavioral data
behavioral_data <- add_sv(behavioral_data)

# =============================================================================
# Load pupil GLM betas
# =============================================================================
message("Loading pupil GLM betas...")
pupil_betas <- read.csv(file.path(analysis_data_path, "preprocessed",
                                  "pupilbetas_glm_preprocessed.csv"))
# Make exclude_startAligned boolean
pupil_betas <- pupil_betas %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned))

# Merge with behavioral data to obtain trial-level regressors
message("Merging with behavioral data...")
data_betas <- merge(pupil_betas,
                    behavioral_data,
                    by = c("sub", "trial", "run", "dataset"))
# Apply pupil exclusion criteria
data_betas_clean <- data_betas %>%
  filter(!exclude_startAligned)

# Scale regressors within-subject (reuses existing pupil regressor pipeline)
data_betas_clean <- create_pupil_regressors(data_betas_clean)

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
# Merge with pupil data
data_start <- merge(pupil_data_start, behavioral_data,
                    by = c("sub", "trial", "dataset"))

data_response <- merge(pupil_data_response, behavioral_data,
                       by = c("sub", "trial", "dataset"))

# =============================================================================
# Apply exclusion criteria
# =============================================================================
message("Applying exclusion criteria...")
# Filter to non-excluded trials
data_start_clean <- data_start %>%
  filter(!pupil_exclude)
data_response_clean <- data_response %>%
  filter(!pupil_exclude)

# Add regressors of interest
message("Creating pupil regressors...")
data_start_clean <- create_pupil_regressors(data_start_clean)
data_response_clean <- create_pupil_regressors(data_response_clean)

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
