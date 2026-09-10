# =============================================================================
# Choice History Under Risk: Config for Accept GLME Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_acceptglme"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

# Path to shared utility functions
shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(broom.mixed)
library(dplyr)
library(tibble)
library(lmerTest)
library(bayestestR)
library(marginaleffects)
library(brglm2)

# Source shared settings and utility functions
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_extractrandomeffects.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))
source(file.path(shared_utils_path, "utils_modelcomparison.R"))
source(file.path(repo_path, "src", "riskhistory", "setup_behavioraldata", "utils_getchoicehistory.R"))

# =============================================================================
# Load and prepare data
# =============================================================================
behavioral_data_full <- load_behavioral_data()
behavioral_data <- subset(behavioral_data_full, !exclude)

# =============================================================================
# Initialize model storage lists
# =============================================================================
behavioral_models_list <- list()
for (ds in datasets) {
  behavioral_models_list[[ds]] <- list()
}

# Print the number of unique participants in each dataset for verification
for (ds in datasets) {
  n_subs <- length(unique(behavioral_data$sub[behavioral_data$dataset == ds]))
  message(sprintf("Dataset: %s, Unique Participants: %d", ds, n_subs))
  # Print the mean number of trials per participant for verification
  mean_trials <- behavioral_data %>%
    filter(dataset == ds) %>%
    group_by(sub) %>%
    summarise(n_trials = n()) %>%
    summarise(mean_trials = mean(n_trials))
  message(sprintf("Dataset: %s, Mean Trials per Participant: %.2f", ds, mean_trials$mean_trials))
}