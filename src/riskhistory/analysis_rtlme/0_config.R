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
# Source utility functions
# =============================================================================
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_createregressors.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# check_model_diagnostics() is defined in analysis_acceptglme
source(file.path(acceptglme_path, "utils_modelanalytics.R"))

# add_transition_regressor() -- same current x previous choice transition
# factor used in analysis_pupiltimeseries / analysis_pupilbetas
pupiltimeseries_path <- file.path(repo_path, "src", "riskhistory",
                                  "analysis_pupiltimeseries")
source(file.path(pupiltimeseries_path, "utils_createtransitionregressor.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)
datasets <- c("equalIndifference", "equalRange", "sheng2020")
p_method <- "BH"

LMER_CONTROL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load and prepare data
# =============================================================================
message("Loading behavioral data...")
behavioral_data_full <- read.csv(file.path(analysis_data_path,
                                           "preprocessed",
                                           "behavioral_data.csv"))
behavioral_data_full <- behavioral_data_full %>%
  mutate(exclude = as.logical(exclude))

# add_subjectivevalues() merges the M1 baseline betas, computes sv, and
# subsets to non-excluded trials
message("Computing subjective values...")
pth_betas <- file.path(results_path, "analysis_acceptglme",
                       "1_baselinebehavior_betas.csv")
rtlme_data <- add_subjectivevalues(df = behavioral_data_full, pth_betas = pth_betas)

rtlme_data$logRT <- log(rtlme_data$rt)
rtlme_data$sv_abs <- abs(rtlme_data$sv)

# 4-level current x previous choice transition factor (accept_to_accept,
# accept_to_reject, reject_to_accept, reject_to_reject). Rows with
# choice_prev1 == 0 (no valid t-1 trial, see NOTE below) get transition = NA
# and are dropped by lm()'s default na.action in any model using this term.
message("Creating transition (current x previous choice) regressor...")
rtlme_data <- add_transition_regressor(rtlme_data)

message("Config loaded successfully.")
for (ds in datasets) {
  n_subs <- length(unique(rtlme_data$sub[rtlme_data$dataset == ds]))
  n_trials <- sum(rtlme_data$dataset == ds)
  message(sprintf("Dataset: %s, Unique Participants: %d, Trials: %d",
                  ds, n_subs, n_trials))
}
