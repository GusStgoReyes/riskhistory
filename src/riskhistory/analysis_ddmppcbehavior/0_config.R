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

message("Loading DDM posterior predictive data...")
ddm_data <- read.csv(file.path(results_path, "analysis_ppcddm",
                               "HDDM_Base_simulated_data.csv"))

ddm_data <- ddm_data %>%
  mutate(
    logit_mean_accept = qlogis(mean_accept)
  )

for (ds in datasets) {
  n_subs <- length(unique(ddm_data$sub[ddm_data$dataset == ds]))
  n_rows <- nrow(ddm_data[ddm_data$dataset == ds, ])
  message(sprintf("Dataset: %s, Subjects: %d, Rows: %d", ds, n_subs, n_rows))
}

message("Config loaded successfully.")
