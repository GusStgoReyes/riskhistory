# =============================================================================
# Shared data-loading functions for R analysis pipelines
# =============================================================================
# Assumes the base config.R and utils_createregressors.R
# have already been sourced by the caller.

# (1) Raw preprocessed behavioral data, filtered to non-excluded trials.
load_behavioral_data <- function() {
  df <- read.csv(behavioral_data_path)
  df$exclude <- as.logical(df$exclude)
  df
}

# (2) Per-timepoint pupil data for one alignment, with baseline subtracted
# and filtered to pupil-clean trials
load_pupil_data <- function(alignment = c("start", "response")) {
  alignment <- match.arg(alignment)
  filename <- if (alignment == "start") {
    "pupil_data_alignedStartTime.csv"
  } else {
    "pupil_data_alignedResponseTime.csv"
  }
  read.csv(file.path(preprocessed_data_path, filename)) %>%
    mutate(pupil_size = pupil_size - baseline_pupil)
}

# (3) Behavioral data merged with per-trial pupil GLM betas, filtered to
# pupil-clean trials (exclude_startAligned covers the behavioral exclusions)
load_behavioral_pupil_data <- function(
  pth_betas = file.path(results_path, "analysis_acceptglme",
                        "1_prospecttheory_baseline_betas.csv")
) {
  behavioral_data <- load_behavioral_data()
  data <- add_subjectivevalues(behavioral_data, pth_betas)

  pupil_betas <- read.csv(file.path(preprocessed_data_path,
                                    "pupilbetas_glm_preprocessed.csv")) %>%
    mutate(exclude_startAligned = as.logical(exclude_startAligned))

  data <- merge(data, pupil_betas,
                by = c("sub", "trial", "run", "dataset"))
  data$exclude <- data$exclude_startAligned
  data <- subset(data, !exclude)
  standardize_behavioral_regressors(data)
}
