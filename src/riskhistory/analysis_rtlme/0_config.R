# =============================================================================
# Choice History Under Risk: Config for RT LME Analysis
# =============================================================================
# Analysis-specific settings for the RT (log response time) LME models.
# Tests whether RT depends on subjective value, current-trial outcome
# (accept/reject), and whether the current trial is a switch from the
# previous choice -- both as an interaction on the full data (Model 1) and
# as a 3-trial choice-sequence factor fit separately on accept trials and
# reject trials (Model 2).
# Paths are imported from the base config.
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_rtlme"
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
library(emmeans)

# =============================================================================
# Source utility functions
# =============================================================================
# Reuse add_subjectivevalues() (sv), extract_fixed_effects(), and the
# convergence-report helpers from analysis_acceptglme rather than duplicating
# them here.
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_createregressors.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# check_model_diagnostics() is defined in analysis_acceptglme
source(file.path(acceptglme_path, "utils_modelanalytics.R"))

source(file.path(utils_path, "utils_createsequence.R"))

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

# RT scales with distance from indifference, not the sign of sv (accept vs
# reject already carries the sign via the `accept`/`sequence` terms), so the
# RT models use |sv|.
rtlme_data$sv_abs <- abs(rtlme_data$sv)

# NOTE: choice_prev1/choice_prev2 == 0 is a "no trial there yet" sentinel
# (see add_exclude_trials() in setup_behavioraldata/utils_behavioraldata.py),
# not a reject choice -- e.g. trial_run == 2 has no valid t-2 trial. These
# trials are intentionally kept here since Model 1 only needs choice_prev1
# (via `switch`), which is valid for all of them. Model 2's sequence factor
# additionally needs a real choice_prev2, so it applies its own filter
# locally in 2_rtlme_sequencemodels.R rather than dropping these trials here.

message("Config loaded successfully.")
for (ds in datasets) {
  n_subs <- length(unique(rtlme_data$sub[rtlme_data$dataset == ds]))
  n_trials <- sum(rtlme_data$dataset == ds)
  message(sprintf("Dataset: %s, Unique Participants: %d, Trials: %d",
                  ds, n_subs, n_trials))
}
