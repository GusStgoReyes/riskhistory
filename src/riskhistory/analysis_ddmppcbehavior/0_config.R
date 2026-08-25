# =============================================================================
# Config for DDM Posterior Predictive Behavior Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_ddmppcbehavior"
analysis_results_path <- file.path(results_path, analysis_name)

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# Path to utility functions (in this directory)
utils_path <- dirname(sys.frame(1)$ofile)
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
pupiltimeseries_path <- file.path(repo_path, "src", "riskhistory", "analysis_pupiltimeseries")

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)
library(broom.mixed)
library(dplyr)
library(tibble)
library(marginaleffects)

# Source utility functions
source(file.path(utils_path, "utils_savebetas.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))
source(file.path(acceptglme_path, "utils_modelanalytics.R"))

# add_transition_regressor() -- same current x previous choice transition
# factor used in analysis_pupiltimeseries / analysis_rtlme
source(file.path(pupiltimeseries_path, "utils_createtransitionregressor.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"  # Benjamini-Hochberg

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
LMER_CONTROL  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

N_PPC_SAMPLES <- 1000

# =============================================================================
# DDM models to analyze (one {model_id}_simulated_data.csv per model, from
# analysis_ppcddm). Split into two groups since they need different
# formulas downstream:
#   BEHAVIOR_DDM_MODELS -- no "pupil" column, used by 1_acceptglme.R /
#     2_rtlme.R.
#   PUPIL_DDM_MODELS -- simulated on the pupil-clean population, carry a
#     real "pupil" column (and "pupil_bin"), used by 3_pupilaccept.R.
#     HDDM_Base_Lag1-stpdrift_Pupil has no pupil *interaction* term in the
#     DDM itself, but is still simulated on the pupil-clean population (so
#     its PPC data has real pupil values too) -- it's the null-comparison
#     baseline against the 5 true pupil-mechanism models.
# =============================================================================
BEHAVIOR_DDM_MODELS <- c(
  "HDDM_Base",
  "HDDM_Base_Lag1-drift",
  "HDDM_Base_Lag1-stp",
  "HDDM_Base_Lag1-stpdrift"
)

PUPIL_DDM_MODELS <- c(
  "HDDM_Base_Lag1-stpdrift_Pupil",
  "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift",
  "HDDM_Base_Lag1-stpdrift_Pupil_valuation",
  "HDDM_Base_Lag1-stpdrift_Pupil_boundary",
  "HDDM_Base_Lag1-stpdrift_Pupil_drift",
  "HDDM_Base_Lag1-stpdrift_Pupil_stp"
)

# Load one DDM model's simulated-data CSV and add logit_mean_accept.
# logit (not the raw [0,1] proportion) since mean_accept is an average of
# N_PPC_SAMPLES simulated binary choices per condition, so it needs an
# unbounded scale for lmer (rather than literally re-fitting a GLMM on
# synthetic Bernoulli draws).
load_ddm_ppc_data <- function(model_id) {
  path <- file.path(results_path, "analysis_ppcddm",
                    paste0(model_id, "_simulated_data.csv"))
  df <- read.csv(path)
  df$logit_mean_accept <- qlogis(df$mean_accept)
  df
}

message("Config loaded successfully.")
