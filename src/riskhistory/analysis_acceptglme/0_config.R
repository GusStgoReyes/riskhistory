# =============================================================================
# Choice History Under Risk: Config for Accept GLME Analysis
# =============================================================================
# This file contains analysis-specific settings for the accept/reject GLME
# analysis. Paths are imported from the base config.
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_acceptglme"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

# Path to utility functions (in this directory)
utils_path <- dirname(sys.frame(1)$ofile)

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(broom.mixed)
library(dplyr)
library(tibble)
library(lmerTest)
library(bayestestR)

# Source utility functions
source(file.path(utils_path, "utils_createregressors.R"))
source(file.path(utils_path, "utils_checkmodelconverge.R"))
source(file.path(utils_path, "utils_saverandomeffectsub.R"))
source(file.path(utils_path, "utils_savefixedeffects.R"))
source(file.path(utils_path, "utils_modelcomparison.R"))
source(file.path(repo_path, "src", "riskhistory", "setup_behavioraldata", "utils_getchoicehistory.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

# Define datasets to analyze
datasets <- c("sheng2020", "equalRange", "equalIndifference")

# Multiple comparisons method
p_method <- "BH"  # Benjamini-Hochberg

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
# =============================================================================
# Load and prepare data
# =============================================================================
# Load preprocessed behavioral data
behavioral_data_full <- read.csv(file.path(analysis_data_path,
                                           "preprocessed",
                                           "behavioral_data.csv"))
behavioral_data_full <- behavioral_data_full %>%
  mutate(exclude = as.logical(exclude))

# Add log RT column
behavioral_data_full$log_rt <- log(behavioral_data_full$rt)

# Filter out excluded trials for behavioral analyses
behavioral_data <- subset(behavioral_data_full, !exclude)

# Fill any residual NAs in choice_prev1 (first trial of each run) with 0
behavioral_data[["choice_prev1"]][is.na(behavioral_data[["choice_prev1"]])] <- 0

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