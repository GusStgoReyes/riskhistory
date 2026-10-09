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

# Path to shared utility functions
shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)
library(broom.mixed)
library(dplyr)

# Source shared settings and utility functions
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))
source(file.path(shared_utils_path, "utils_modelanalytics.R"))

# =============================================================================
# Analysis settings
# =============================================================================
N_PPC_SAMPLES <- 1000

load_ddm_ppc_data <- function(model_id) {
  path <- file.path(results_path, "analysis_ppcddm",
                    paste0(model_id, "_simulated_data_sampled.csv"))
  df <- read.csv(path)
  eps <- 1 / (2 * N_PPC_SAMPLES)
  mean_accept_clipped <- pmin(pmax(df$mean_accept, eps), 1 - eps)
  df$logit_mean_accept <- qlogis(mean_accept_clipped)
  df
}

save_csv <- function(df, filename) {
  write.csv(df, file.path(analysis_results_path, filename), row.names = FALSE)
  message("Saved: ", filename)
}

message("Config loaded successfully.")
