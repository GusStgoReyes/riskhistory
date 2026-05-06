# =============================================================================
# Choice History Under Risk: Config for Pupil-Behavior Analysis
# =============================================================================
# Analysis-specific settings for pupil-behavior GLME models.
# Tests whether pupil betas (trial-by-trial decision period amplitude) predict
# choice, interact with stimulus values, or moderate choice history bias.
# Paths are imported from the base config.
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_pupilbehavior"
analysis_data_path    <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

utils_path <- dirname(sys.frame(1)$ofile)

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
library(bayestestR)

# Source utility functions
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_modelcomparison.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))

# check_model_diagnostics is defined in analysis_pupillme
pupillme_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_pupillme")
source(file.path(pupillme_utils_path, "utils_modelanalytics.R"))

source(file.path(utils_path, "utils_createregressors.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)
datasets <- c("equalIndifference", "equalRange", "sheng2020")
p_method <- "BH"

# =============================================================================
# Load behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))

# =============================================================================
# Adding the pupil betas
# =============================================================================
message("Pupil betas including...")
# Use the same exclude flag written by setup_pupilbetas
pupilbetas_preprocessed <- read.csv(
  file.path(analysis_data_path, "preprocessed", "pupilbetas_glm_preprocessed.csv")
)
pupilbetas_preprocessed <- pupilbetas_preprocessed %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned)) %>%
  select(sub, run, trial, dataset, exclude_startAligned, pupil)

# Add pupil_prev to pupilbetas_preprocessed
pupilbetas_preprocessed <- pupilbetas_preprocessed %>%
  group_by(sub, run, dataset) %>%
  arrange(trial, .by_group = TRUE) %>%
  mutate(pupil_prev = lag(pupil)) %>%
  ungroup()

data_merged <- merge(behavioral_data, pupilbetas_preprocessed,
                     by = c("sub", "run", "trial", "dataset"))

data_clean <- data_merged %>%
  filter(!exclude_startAligned)

message("Creating regressors...")
data_clean <- create_pupilbehavior_regressors(data_clean)

message(sprintf("Data after exclusion: %d trials", nrow(data_clean)))
for (ds in datasets) {
  n <- nrow(data_clean[data_clean$dataset == ds, ])
  message(sprintf("  %s: %d trials", ds, n))
}

message("Config loaded successfully.")
