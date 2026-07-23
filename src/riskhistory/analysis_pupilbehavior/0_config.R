# =============================================================================
# Choice History Under Risk: Config for Pupil-Behavior Analysis
# =============================================================================
# Analysis-specific settings for pupil-behavior GLME models.
# Tests whether pupil betas (trial-by-trial decision period amplitude) predict
# choice, interact with stimulus values, or moderate choice history bias.
# Paths are imported from the base config.
# =============================================================================

source(file.path(dirname(sys.frame(1)$ofile), "..", "config.R"))

# =============================================================================
# Analysis-specific paths
# =============================================================================
analysis_name         <- "analysis_pupilbehavior"
analysis_data_path    <- file.path(repo_path, "data")
analysis_results_path <- file.path(results_path, analysis_name)

utils_path <- dirname(sys.frame(1)$ofile)

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
library(bayestestR)
library(emmeans)

# Source utility functions
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_modelcomparison.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))
source(file.path(acceptglme_path, "utils_checkmodelconverge.R"))
source(file.path(acceptglme_path, "utils_saverandomeffectsub.R"))

# check_model_diagnostics() is defined in analysis_acceptglme
source(file.path(acceptglme_path, "utils_modelanalytics.R"))

source(file.path(utils_path, "utils_createregressors.R"))

# =============================================================================
# Analysis settings
# =============================================================================
set.seed(42)
datasets <- c("equalIndifference", "equalRange", "sheng2020")
p_method <- "BH"
GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# =============================================================================
# Load behavioral data
# =============================================================================
message("Loading behavioral data...")
behavioral_data <- read.csv(file.path(analysis_data_path,
                                      "preprocessed",
                                      "behavioral_data.csv"))

# =============================================================================
# Adding the residualized pupil
# =============================================================================
message("Loading residualized pupil...")
# results/analysis_pupilbetas/2_pupilbetas_fullmodel_residuals.csv --
# residuals of pupil ~ logRT_z + gain_z + loss_z + entropy_c + abs_delta_ev_z +
# (1 | sub), i.e. pupil dilation with RT/value/entropy and each subject's own
# baseline already regressed out. Behavioral and pupil-specific exclusions
# were already applied upstream when that model was fit (analysis_pupilbetas
# /0_config.R), so an inner merge here restricts data_clean to that same
# already-excluded trial set -- no additional filtering needed.
pupil_residuals <- read.csv(
  file.path(results_path, "analysis_pupilbetas",
            "2_pupilbetas_fullmodel_residuals.csv")
) %>%
  select(sub, run, trial, dataset, pupil, pupil_residual)

data_clean <- merge(behavioral_data, pupil_residuals,
                    by = c("sub", "run", "trial", "dataset"))

message("Creating regressors...")
data_clean <- create_pupilbehavior_regressors(data_clean)

message(sprintf("Data after exclusion: %d trials", nrow(data_clean)))
for (ds in datasets) {
  n <- nrow(data_clean[data_clean$dataset == ds, ])
  message(sprintf("  %s: %d trials", ds, n))
}

message("Config loaded successfully.")
