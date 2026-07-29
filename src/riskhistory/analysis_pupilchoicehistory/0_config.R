# =============================================================================
# Config for Pupil Choice-History Analysis
# =============================================================================
# Tests whether a subject's individual choice-history DDM biases (from
# HDDM_Base_Lag1-full) predict their "pupil switch" signature -- how much
# their pupil beta differs between switch trials (accept_to_reject,
# reject_to_accept) and repeat trials (accept_to_accept, reject_to_reject).
# See utils_pupiltransitiondf.R and 1_pupil_switch_ch.R.
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupilchoicehistory"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

utils_path <- dirname(sys.frame(1)$ofile)
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(dplyr)
library(tidyr)
library(tibble)
library(broom)

# add_transition_regressor() -- same current x previous choice transition
# factor used in analysis_pupiltimeseries / analysis_pupilbetas
pupiltimeseries_path <- file.path(repo_path, "src", "riskhistory",
                                  "analysis_pupiltimeseries")
source(file.path(pupiltimeseries_path, "utils_createtransitionregressor.R"))
source(file.path(utils_path, "utils_pupiltransitiondf.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"

# =============================================================================
# Load behavioral data and apply behavioral exclusion criteria (RT<0.2s/
# >3.8s, <3 valid prior trials, missing gain/loss pairs -- see CLAUDE.md)
# =============================================================================
message("Loading behavioral data...")
behavioral_data_raw <- read.csv(file.path(analysis_data_path,
                                          "preprocessed",
                                          "behavioral_data.csv"))
behavioral_data_raw <- behavioral_data_raw %>%
  mutate(exclude = as.logical(exclude))

behavioral_data <- behavioral_data_raw %>%
  filter(!exclude)

message(sprintf("Behavioral data: %d rows total, %d rows after exclusion",
                nrow(behavioral_data_raw), nrow(behavioral_data)))

mean_accept_rate <- behavioral_data %>%
  group_by(dataset, sub) %>%
  summarise(mean_accept_rate = mean(accept, na.rm = TRUE), .groups = "drop")

# =============================================================================
# Load pupil GLM betas (carries both "pupil" -- raw GLM amplitude -- and
# "pupil_residual", appended by analysis_pupilbetas/2_pupilbetas_
# regressout.R) and merge with (already behaviorally-excluded) behavioral data
# =============================================================================
message("Loading pupil GLM betas...")
pupil_betas <- read.csv(file.path(analysis_data_path, "preprocessed",
                                  "pupilbetas_glm_preprocessed.csv"))
pupil_betas <- pupil_betas %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned))

message("Merging behavioral and pupil betas data...")
data_betas <- merge(pupil_betas, behavioral_data,
                    by = c("sub", "trial", "run", "dataset"))

message("Applying pupil exclusion criteria...")
data_betas_clean <- data_betas %>%
  filter(!exclude_startAligned)

message(sprintf("Pupil betas data: %d rows total, %d rows after exclusion",
                nrow(data_betas), nrow(data_betas_clean)))

# =============================================================================
# Choice-history transition regressor
# =============================================================================
message("Creating transition (current x previous choice) regressor...")
data_betas_clean <- add_transition_regressor(data_betas_clean)

message("Config loaded successfully.")
