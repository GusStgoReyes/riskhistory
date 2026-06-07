# =============================================================================
# Config for Pupil Timeseries Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupiltimeseries"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

utils_path      <- dirname(sys.frame(1)$ofile)
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")

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
library(lmerTest)

# Source utility functions from analysis_acceptglme
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_createregressors.R"))
source(file.path(acceptglme_path, "utils_modelcomparison.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"  # Benjamini-Hochberg

LMER_CONTROL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load and prepare behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))
behavioral_data <- behavioral_data %>%
  mutate(exclude = as.logical(exclude))

# Merge betas, compute sv/entropy, and apply behavioral exclusion
message("Loading subject-specific betas and computing subjective value...")
betas_path <- file.path(results_path, "analysis_acceptglme",
                        "1_baselinebehavior_betas.csv")
behavioral_data <- add_subjectivevalues(behavioral_data, betas_path)

# =============================================================================
# Load pupil data
# =============================================================================
message("Loading pupil data...")

pupil_data_start <- read.csv(file.path(analysis_data_path,
                                       "preprocessed",
                                       "pupil_data_alignedStartTime.csv"))

pupil_data_response <- read.csv(file.path(analysis_data_path,
                                          "preprocessed",
                                          "pupil_data_alignedResponseTime.csv"))

# Remove baseline from pupil size
pupil_data_start <- pupil_data_start %>%
  mutate(pupil_size = pupil_size - baseline_pupil)

pupil_data_response <- pupil_data_response %>%
  mutate(pupil_size = pupil_size - baseline_pupil)

# Rename exclude column in pupil data to avoid conflict
pupil_data_start <- pupil_data_start %>%
  rename(pupil_exclude = exclude)

pupil_data_response <- pupil_data_response %>%
  rename(pupil_exclude = exclude)

# =============================================================================
# Merge and apply exclusion criteria
# =============================================================================
message("Merging behavioral and pupil data...")

data_start <- merge(pupil_data_start, behavioral_data,
                    by = c("sub", "trial", "dataset"))

data_response <- merge(pupil_data_response, behavioral_data,
                       by = c("sub", "trial", "dataset"))

message("Applying pupil exclusion criteria...")
data_start_clean <- data_start %>% filter(!pupil_exclude)
data_response_clean <- data_response %>% filter(!pupil_exclude)

message(sprintf("Start-aligned:    %d rows total, %d rows after exclusion",
                nrow(data_start), nrow(data_start_clean)))
message(sprintf("Response-aligned: %d rows total, %d rows after exclusion",
                nrow(data_response), nrow(data_response_clean)))

for (ds in datasets) {
  ds_data <- data_start_clean %>% filter(dataset == ds)
  sub_summary <- ds_data %>%
    group_by(sub) %>%
    summarise(
      n_trials    = n_distinct(trial),
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
data_start_clean    <- create_pupil_regressors(data_start_clean)
data_response_clean <- create_pupil_regressors(data_response_clean)

# =============================================================================
# Get unique timebins
# =============================================================================
timebins_start    <- sort(unique(data_start_clean$timebin))
timebins_response <- sort(unique(data_response_clean$timebin))

message(sprintf("Start-aligned timebins:    %d (%.2f to %.2f s)",
                length(timebins_start),
                min(timebins_start), max(timebins_start)))
message(sprintf("Response-aligned timebins: %d (%.2f to %.2f s)",
                length(timebins_response),
                min(timebins_response), max(timebins_response)))

message("Config loaded successfully.")
