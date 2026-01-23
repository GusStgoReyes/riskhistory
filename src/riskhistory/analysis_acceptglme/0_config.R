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
source(file.path(utils_path, "utils_addsubjectivevalues.R"))
source(file.path(utils_path, "utils_checkmodelconverge.R"))
source(file.path(utils_path, "utils_saverandomeffectsub.R"))
source(file.path(utils_path, "utils_savefixedeffects.R"))
source(file.path(utils_path, "utils_modelcomparison.R"))
source(file.path(utils_path, "utils_scalestimuli.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

# Define datasets to analyze
datasets <- c("equalIndifference", "equalRange", "sheng2020")

# Multiple comparisons method
p_method <- "BH"  # Benjamini-Hochberg

# =============================================================================
# Load and prepare data
# =============================================================================
# Load preprocessed behavioral data
behavioral_data_full <- read.csv(file.path(analysis_data_path,
                                           "preprocessed",
                                           "behavioral_data.csv"))

# Add log RT column
behavioral_data_full$log_rt <- log(behavioral_data_full$rt)

# Apply scaling function to the behavioral data
behavioral_data_full <- scale_stimuli(behavioral_data_full)

# Filter out excluded trials for behavioral analyses
behavioral_data <- subset(behavioral_data_full, exclude == "False")

# =============================================================================
# Initialize model storage lists
# =============================================================================
behavioral_models_list <- list()
for (ds in datasets) {
  behavioral_models_list[[ds]] <- list()
}

rt_models_list <- list()
for (ds in datasets) {
  rt_models_list[[ds]] <- list()
}