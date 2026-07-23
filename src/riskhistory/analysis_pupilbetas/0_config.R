# =============================================================================
# Config for Pupil Betas Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupilbetas"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

# Path to utility functions
utils_path <- dirname(sys.frame(1)$ofile)
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")

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
library(lmtest)
library(lmerTest)
library(bayestestR)
library(marginaleffects)

# Source utility functions from analysis_acceptglme
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_createregressors.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# add_transition_regressor() -- same current x previous choice transition
# factor used in analysis_pupiltimeseries
pupiltimeseries_path <- file.path(repo_path, "src", "riskhistory",
                                  "analysis_pupiltimeseries")
source(file.path(pupiltimeseries_path, "utils_createtransitionregressor.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"  # Benjamini-Hochberg

LMER_CONTROL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load and prepare behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))
behavioral_data <- behavioral_data %>%
  mutate(exclude = as.logical(exclude))

# Merge betas, compute sv/entropy, and apply behavioral exclusion
message("Loading subject-specific betas and computing subjective value...")
betas_path <- file.path(results_path, "analysis_acceptglme",
                        "1_baselinebehavior_betas.csv")
behavioral_data <- add_subjectivevalues(behavioral_data, betas_path)

# =============================================================================
# Load pupil GLM betas
# =============================================================================
message("Loading pupil GLM betas...")
pupil_betas <- read.csv(file.path(analysis_data_path, "preprocessed",
                                  "pupilbetas_glm_preprocessed.csv"))
pupil_betas <- pupil_betas %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned))

# =============================================================================
# Merge and apply exclusion criteria
# =============================================================================
message("Merging behavioral and pupil betas data...")
data_betas <- merge(pupil_betas, behavioral_data,
                    by = c("sub", "trial", "run", "dataset"))

message("Applying pupil exclusion criteria...")
data_betas_clean <- data_betas %>%
  filter(!exclude_startAligned)

message(sprintf("Pupil betas data: %d rows total, %d rows after exclusion",
                nrow(data_betas), nrow(data_betas_clean)))
for (ds in datasets) {
  ds_data <- data_betas_clean %>% filter(dataset == ds)
  sub_summary <- ds_data %>%
    group_by(sub) %>%
    summarise(
      n_trials   = n(),
      mean_accept = mean(accept, na.rm = TRUE),
      mean_rt     = mean(rt,     na.rm = TRUE),
      .groups = "drop"
    )
  n_subs      <- nrow(sub_summary)
  mean_trials <- mean(sub_summary$n_trials)
  mean_accept <- mean(sub_summary$mean_accept)
  mean_rt     <- mean(sub_summary$mean_rt)
  message(sprintf("  %s: %d subjects, %.1f trials/subject, %.2f mean accept, %.3f mean RT",
                  ds, n_subs, mean_trials, mean_accept, mean_rt))
}

# =============================================================================
# Create scaled within-subject regressors
# =============================================================================
message("Creating pupil regressors...")
data_betas_clean <- create_pupil_regressors(data_betas_clean)

message("Creating transition (current x previous choice) regressor...")
data_betas_clean <- add_transition_regressor(data_betas_clean)

message("Config loaded successfully.")
