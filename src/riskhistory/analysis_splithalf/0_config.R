# =============================================================================
# Choice History Under Risk: Config for Split-Half Analysis
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_splithalf"
analysis_results_path <- file.path(results_path, analysis_name)

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(dplyr)

# =============================================================================
# Source shared settings and utility functions
# =============================================================================
shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_rtswitch.R"))

# =============================================================================
# Split halves (same as utils.py::HALF_BY_RUN)
# =============================================================================
add_half_id <- function(df) {
  df$half_id <- ifelse(df$dataset == "sheng2020", df$run,
                       ifelse(df$run %in% c(1, 4), 1, 2))
  df
}

# =============================================================================
# Load and prepare data
# =============================================================================
message("Loading behavioral data...")
behavioral_data_full <- add_half_id(load_behavioral_data())

message("Computing half-specific subjective values...")
pth_betas <- file.path(analysis_results_path,
                       "1_prospecttheory_baseline_betas_splithalf.csv")
splithalf_data <- bind_rows(lapply(sort(unique(behavioral_data_full$half_id)), function(h) {
  add_subjectivevalues_and_regressors(
    df = behavioral_data_full[behavioral_data_full$half_id == h, ],
    pth_betas = pth_betas,
    by = c("sub", "dataset", "half_id")
  )
}))

message("Config loaded successfully.")
for (ds in datasets) {
  for (h in sort(unique(splithalf_data$half_id))) {
    df_h <- splithalf_data[splithalf_data$dataset == ds & splithalf_data$half_id == h, ]
    message(sprintf("Dataset: %s, Half: %d, Unique Participants: %d, Trials: %d",
                    ds, h, length(unique(df_h$sub)), nrow(df_h)))
  }
}
