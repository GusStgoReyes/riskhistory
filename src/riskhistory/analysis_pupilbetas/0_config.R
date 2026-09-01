# =============================================================================
# Config for Pupil Betas Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupilbetas"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

# Path to shared utility functions
shared_utils_path <- file.path(repo_path, "src", "riskhistory", "analysis_utils")

# Create results directory if it doesn't exist
if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(broom.mixed)
library(dplyr)
library(tibble)
library(lmtest)
library(lmerTest)
library(bayestestR)
library(marginaleffects)

# Source shared settings and utility functions
source(file.path(shared_utils_path, "0_config.R"))
source(file.path(shared_utils_path, "utils_data.R"))
source(file.path(shared_utils_path, "utils_checkmodelconverge.R"))
source(file.path(shared_utils_path, "utils_createregressors.R"))
source(file.path(shared_utils_path, "utils_savefixedeffects.R"))

# add_transition_regressor() -- same current x previous choice transition
# factor used in analysis_pupiltimeseries
source(file.path(shared_utils_path, "utils_createtransitionregressor.R"))

# =============================================================================
# Load and prepare behavioral + pupil data
# =============================================================================
# load_behavioral_pupil_data() merges the M1 baseline betas, computes sv, and
# merges in the pupil GLM betas, filtered to behavioral- and pupil-clean trials.
message("Loading behavioral and pupil betas data...")
data_betas_clean <- load_behavioral_pupil_data()

# Per-subject mean acceptance rate, computed after both the behavioral and
# pupil exclusions above -- broadcast onto every row for that sub/dataset.
data_betas_clean <- data_betas_clean %>%
  group_by(sub, dataset) %>%
  mutate(mean_accept = mean(accept, na.rm = TRUE)) %>%
  ungroup()

message(sprintf("Pupil betas data: %d rows after exclusion", nrow(data_betas_clean)))
for (ds in datasets) {
  ds_data <- data_betas_clean %>% filter(dataset == ds)
  sub_summary <- ds_data %>%
    group_by(sub) %>%
    summarise(
      n_trials   = n(),
      mean_accept = mean(accept, na.rm = TRUE),
      mean_rt     = mean(rt,     na.rm = TRUE),
      .groups = "drop"
    )
  n_subs      <- nrow(sub_summary)
  mean_trials <- mean(sub_summary$n_trials)
  mean_accept <- mean(sub_summary$mean_accept)
  mean_rt     <- mean(sub_summary$mean_rt)
  message(sprintf("  %s: %d subjects, %.1f trials/subject, %.2f mean accept, %.3f mean RT",
                  ds, n_subs, mean_trials, mean_accept, mean_rt))
}

# =============================================================================
# Create scaled within-subject regressors
# =============================================================================
message("Creating pupil regressors...")
data_betas_clean <- standardize_pupil_regressors(data_betas_clean)

message("Creating transition (current x previous choice) regressor...")
data_betas_clean <- add_transition_regressor(data_betas_clean)

message("Creating choice alteration regressor...")
data_betas_clean <- data_betas_clean %>%
  mutate(
    choice_prev1_01 = ifelse(choice_prev1 == 0, NA_real_, (choice_prev1 + 1) / 2),
    choice_prev2_01 = ifelse(choice_prev2 == 0, NA_real_, (choice_prev2 + 1) / 2),
    # Continuous version (0, 0.5, 1) for the fixed-effects model.
    choice_alteration_num = 2*(abs(accept - (choice_prev1_01 + choice_prev2_01) / 2)- 0.5),
    # Categorical version (Low/Medium/High) for avg_predictions/avg_comparisons.
    choice_alteration = case_when(
      is.na(choice_alteration_num) ~ NA_character_,
      choice_alteration_num == -1 ~ "Low",
      choice_alteration_num == 1 ~ "High",
      TRUE ~ "Medium"
    ),
    choice_alteration = factor(choice_alteration, levels = c("Low", "Medium", "High"))
  ) %>%
  select(-choice_prev1_01, -choice_prev2_01)

n_na_alteration <- sum(is.na(data_betas_clean$choice_alteration))
message(sprintf("  %d row(s) have NA choice_alteration (fewer than 2 valid previous trials in that run)",
                n_na_alteration))

data_betas_clean <- data_betas_clean %>%
  mutate(
    choice_alteration_accept = factor(
      ifelse(accept == 1, as.character(choice_alteration), "Low"),
      levels = c("Low", "Medium", "High")
    ),
    choice_alteration_reject = factor(
      ifelse(accept == 0, as.character(choice_alteration), "Low"),
      levels = c("Low", "Medium", "High")
    ),
    choice_alteration_accept_num = ifelse(accept == 1, choice_alteration_num, 0),
    choice_alteration_reject_num = ifelse(accept == 0, choice_alteration_num, 0),
    # 6-category grouping label for avg_predictions() -- not a model term,
    # just used to group predictions by (choice_alteration x accept).
    choice_alteration_combo = case_when(
      is.na(choice_alteration) ~ NA_character_,
      accept == 1 ~ paste0(as.character(choice_alteration), "_Accept"),
      TRUE ~ paste0(as.character(choice_alteration), "_Reject")
    )
  )

message("Config loaded successfully.")
