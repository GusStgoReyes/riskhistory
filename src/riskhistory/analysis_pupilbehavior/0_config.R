# =============================================================================
# Config for Pupil-Behavior Analysis
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name <- "analysis_pupilbehavior"
analysis_data_path <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

utils_path <- dirname(sys.frame(1)$ofile)
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")

if (!dir.exists(analysis_results_path)) {
  dir.create(analysis_results_path, recursive = TRUE)
}

# =============================================================================
# Load required packages
# =============================================================================
library(lme4)
library(lmerTest)
library(broom.mixed)
library(dplyr)
library(tibble)
library(marginaleffects)

# Source utility functions from analysis_acceptglme
source(file.path(acceptglme_path, "utils_createregressors.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))
source(file.path(acceptglme_path, "utils_modelanalytics.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"  # Benjamini-Hochberg

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
LMER_CONTROL  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load and prepare behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data_full <- read.csv(file.path(analysis_data_path,
                                           "preprocessed",
                                           "behavioral_data.csv"))
behavioral_data_full <- behavioral_data_full %>%
  mutate(exclude = as.logical(exclude))

# add_subjectivevalues() merges the M1 baseline betas, computes sv, and
# subsets to non-excluded trials (behavioral exclusion criteria)
message("Computing subjective values...")
pth_betas <- file.path(results_path, "analysis_acceptglme",
                       "1_baselinebehavior_betas.csv")
behavioral_data <- add_subjectivevalues(df = behavioral_data_full, pth_betas = pth_betas)

behavioral_data$logRT  <- log(behavioral_data$rt)
behavioral_data$sv_abs <- abs(behavioral_data$sv)

# =============================================================================
# Load pupil GLM betas
# =============================================================================
message("Loading pupil GLM betas...")
pupil_betas <- read.csv(file.path(analysis_data_path, "preprocessed",
                                  "pupilbetas_glm_preprocessed.csv"))
pupil_betas <- pupil_betas %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned))

# =============================================================================
# Merge and apply pupil exclusion criteria
# (behavioral exclusion criteria were already applied by add_subjectivevalues())
# =============================================================================
message("Merging behavioral and pupil betas data...")
data_pupilbehavior <- merge(behavioral_data, pupil_betas,
                            by = c("sub", "trial", "run", "dataset"))

message("Applying pupil exclusion criteria...")
data_pupilbehavior <- data_pupilbehavior %>%
  filter(!exclude_startAligned)

# Switch dummies split by the direction of the switch (same
# accept_to_accept/accept_to_reject/reject_to_accept/reject_to_reject
# transitions as analysis_pupiltimeseries/utils_createtransitionregressor.R):
# switch_AA_RA is 1 for reject-to-accept (switched into accept), 0 for
# accept-to-accept (repeated into accept); switch_RR_AR is 1 for
# accept-to-reject (switched into reject), 0 for reject-to-reject (repeated
# into reject). Repeat trials (AA, RR) are 0 on both, the same shared
# reference level the single `switch` regressor already collapsed them into.
data_pupilbehavior <- data_pupilbehavior %>%
  mutate(
    switch_AA_RA = ifelse(accept == 1 & choice_prev1 == -1, 1, 0),
    switch_RR_AR = ifelse(accept == 0 & choice_prev1 == 1, 1, 0)
  )

message(sprintf("Pupil-behavior data: %d rows after behavioral + pupil exclusion",
                nrow(data_pupilbehavior)))
for (ds in datasets) {
  ds_data <- data_pupilbehavior %>% filter(dataset == ds)
  n_subs   <- length(unique(ds_data$sub))
  n_trials <- nrow(ds_data)
  message(sprintf("  %s: %d subjects, %d trials", ds, n_subs, n_trials))
}

message("Config loaded successfully.")
