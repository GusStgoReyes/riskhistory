# =============================================================================
# Shared data-loading functions for R analysis pipelines
# =============================================================================
# Assumes the base config.R (repo_path, preprocessed_data_path,
# behavioral_data_path, results_path) and utils_createregressors.R
# (add_subjectivevalues, standardize_behavioral_regressors,
# add_subjectivevalues_and_regressors) have already been sourced by the
# caller.

# (1) Raw preprocessed behavioral data, filtered to non-excluded trials.
load_behavioral_data <- function() {
  df <- read.csv(behavioral_data_path)
  df$exclude <- as.logical(df$exclude)
  df
}

# (2) Per-timepoint pupil data for one alignment, with baseline subtracted
# and filtered to pupil-clean trials (its own exclude column is renamed to
# pupil_exclude first, to avoid colliding with the behavioral exclude column
# after merging with behavioral data).
load_pupil_data <- function(alignment = c("start", "response")) {
  alignment <- match.arg(alignment)
  filename <- if (alignment == "start") {
    "pupil_data_alignedStartTime.csv"
  } else {
    "pupil_data_alignedResponseTime.csv"
  }
  df <- read.csv(file.path(preprocessed_data_path, filename))
  df <- df %>%
    mutate(pupil_size = pupil_size - baseline_pupil) %>%
    rename(pupil_exclude = exclude)
  df$pupil_exclude <- as.logical(df$pupil_exclude)
}

# (3) Behavioral data (with subjective values and behavioral regressors, via
# add_subjectivevalues_and_regressors()) merged with per-trial pupil GLM
# betas, filtered to pupil-clean trials.
load_behavioral_pupil_data <- function(
  pth_betas = file.path(results_path, "analysis_acceptglme",
                        "1_baselinebehavior_betas.csv")
) {
  behavioral_data <- load_behavioral_data()
  behavioral_data <- add_subjectivevalues_and_regressors(behavioral_data, pth_betas)

  pupil_betas <- read.csv(file.path(preprocessed_data_path,
                                    "pupilbetas_glm_preprocessed.csv")) %>%
    mutate(exclude_startAligned = as.logical(exclude_startAligned))

  data <- merge(behavioral_data, pupil_betas, by = c("sub", "trial", "run", "dataset"))
}
