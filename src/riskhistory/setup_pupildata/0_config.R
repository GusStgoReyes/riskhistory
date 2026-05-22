# =============================================================================
# Pupil Data Preprocessing Pipeline: Configuration
# =============================================================================

# Source the base config to get repo_path and standard paths
source(file.path("..", "config.R"))

# Input Paths
narps_input_dir <- file.path(raw_data_path,
                             "narps",
                             "pupil",
                             "asc")
sheng2020_input_file <- file.path(raw_data_path,
                                  "sheng2020",
                                  "pupil",
                                  "eye.csv")

# Output Paths
pupil_output_dir <- file.path(derivatives_path, "pupil")
narps_output_dir <- file.path(pupil_output_dir, "narps")
sheng2020_output_dir <- file.path(pupil_output_dir, "sheng2020")

# Final combined output files
output_start_aligned <- file.path(preprocessed_data_path,
                                  "pupil_data_alignedStartTime.csv")
output_response_aligned <- file.path(preprocessed_data_path,
                                     "pupil_data_alignedResponseTime.csv")
output_statistics <- file.path(results_path,
                               "setup_pupildata",
                               "pupil_data_statistics.csv")
output_summary_statistics <- file.path(results_path,
                                       "setup_pupildata",
                                       "4_summary_statistics.csv")

# Behavioral data (source of behavioral exclusion flags)
behavioral_data_path <- file.path(preprocessed_data_path,
                                  "behavioral_data.csv")

# =============================================================================
# Processing Parameters
# =============================================================================
# Sampling rates
NARPS_SAMPLING_RATE <- c(500, 1000)  # Hz (will be read from ASC files)
SHENG2020_SAMPLING_RATE <- 120  # Hz

# Final output sampling rate (after binning)
OUTPUT_SAMPLING_RATE <- 8  # Hz

# Low-pass filter parameters
LOWPASS_WP <- 4  # Passband frequency (Hz)
LOWPASS_WS <- 8  # Stopband frequency (Hz)
LOWPASS_RP <- 1  # Passband ripple (dB)
LOWPASS_RS <- 35  # Stopband attenuation (dB)

# Epoching parameters
START_EPOCH_LIMITS <- c(0, 4)  # seconds relative to trial start
RESPONSE_EPOCH_LIMITS <- c(-2, 2)  # seconds relative to response
BASELINE_PERIOD <- c(-0.5, 0)  # seconds relative to event

# Statistics calculation windows (for data quality assessment)
STATS_BASELINE_WINDOW <- c(0, 0.5)  # seconds
START_STATS_TASK_WINDOW <- c(0.5, 4.5)  # seconds
STATS_EPOCH_LIMITS <- c(-0.5, 4.5)  # for calculating statistics before filtering

# Exclusion thresholds
MAX_NA_PERCENTAGE <- 90  # Skip subjects/runs with >90% missing data

# Detransient parameters
DETRANSIENT_N <- 16
DETRANSIENT_MAD_THRESH <- NULL

# Deblink parameters (NARPS only)
DEBLINK_EXTEND <- 100  # ms

# NARPS: 64 trials per run, 4 runs
NARPS_TRIALS_PER_RUN <- 64
NARPS_TOTAL_TRIALS <- NARPS_TRIALS_PER_RUN * 4
NARPS_EVENT_START_PATTERN <- "flag_TrialStart*"
NARPS_EVENT_RESPONSE_PATTERN <- "flag_Response*"

# Sheng2020: 100 trials per run, 2 runs
SHENG2020_TRIALS_PER_RUN <- 100
SHENG2020_TOTAL_TRIALS <- SHENG2020_TRIALS_PER_RUN * 2
SHENG2020_EVENT_START_PATTERN <- "Start*"
SHENG2020_EVENT_RESPONSE_PATTERN <- "Response*"


# Manual exclusion subjects NARPS
narps_invalid_subs <- c()

# Manual exclusion subjects Sheng2020
sheng2020_invalid_subs <- c(29)

# =============================================================================
# Helper function to ensure directories exist
# =============================================================================
ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}
