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

BEHAVIOR_DDM_MODELS <- c(
  "HDDM_Base",
  "HDDM_Base_Lag1-drift",
  "HDDM_Base_Lag1-stp",
  "HDDM_Base_Lag1-stpdrift"
)

PUPIL_DDM_MODELS <- c(
  "HDDM_Base_Lag1-stp_Pupil_stp",
  "HDDM_Base_Lag1-drift_Pupil_drift",
  "HDDM_Base_Lag1-drift_Pupil",
  "HDDM_Base_Lag1-stp_Pupil"
)

load_ddm_ppc_data <- function(model_id) {
  path <- file.path(results_path, "analysis_ppcddm",
                    paste0(model_id, "_simulated_data.csv"))
  df <- read.csv(path)
  eps <- 1 / (2 * N_PPC_SAMPLES)
  mean_accept_clipped <- pmin(pmax(df$mean_accept, eps), 1 - eps)
  df$logit_mean_accept <- qlogis(mean_accept_clipped)
  df
}

message("Config loaded successfully.")
