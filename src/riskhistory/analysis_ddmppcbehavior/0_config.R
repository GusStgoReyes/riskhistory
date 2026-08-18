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

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)
library(broom.mixed)
library(dplyr)
library(tibble)

# Source utility functions
source(file.path(utils_path, "utils_savebetas.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"  # Benjamini-Hochberg

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load DDM posterior predictive data and add choice-history columns
# =============================================================================
message("Loading DDM posterior predictive data...")
ddm_data <- read.csv(file.path(results_path, "analysis_ppcddm",
                               "HDDM_Base_simulated_data.csv"))

ddm_data <- ddm_data %>%
  mutate(
    accept = as.integer(mean_accept > 0.5),
    repeat_ = as.integer(pred_repeat > 0.5),
    choice_prev1 = ifelse(accept == repeat_, 1, -1)
  )

for (ds in datasets) {
  n_subs <- length(unique(ddm_data$sub[ddm_data$dataset == ds]))
  n_rows <- nrow(ddm_data[ddm_data$dataset == ds, ])
  message(sprintf("Dataset: %s, Subjects: %d, Rows: %d", ds, n_subs, n_rows))
}

message("Config loaded successfully.")
