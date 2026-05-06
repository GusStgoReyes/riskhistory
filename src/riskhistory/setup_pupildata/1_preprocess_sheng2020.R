# =============================================================================
# Pupil Data Preprocessing Pipeline: Sheng2020 Dataset
# =============================================================================
# Preprocesses Sheng2020 pupil data (CSV format) and saves intermediate outputs.
# This dataset requires conversion to eyeris format before processing.
# =============================================================================

library(eyeris)
library(dplyr)
library(stringr)

script_dir <- local({
  for (i in seq(sys.nframe(), 1, by = -1)) {
    f <- sys.frame(i)$ofile
    if (!is.null(f)) return(dirname(normalizePath(f, mustWork = FALSE)))
  }
  args <- commandArgs(trailingOnly = FALSE)
  flag <- grep("--file=", args, value = TRUE)
  if (length(flag)) dirname(normalizePath(sub("--file=", "", flag))) else getwd()
})
source(file.path(script_dir, "0_config.R"))
source(file.path(script_dir, "utils_eyeris.R"))
source(file.path(script_dir, "utils_sheng2020.R"))

# =============================================================================
# Setup
# =============================================================================

cat("=== Sheng2020 Pupil Data Preprocessing ===\n")
cat("Input file:", sheng2020_input_file, "\n")
cat("Output directory:", sheng2020_output_dir, "\n\n")

ensure_dir(sheng2020_output_dir)

cat("Loading Sheng2020 data...\n")
all_data  <- load_sheng2020_data(sheng2020_input_file)
cat("Loaded", nrow(all_data), "rows\n")

data_list <- split_sheng2020_by_subject_run(all_data)
cat("Found", length(data_list), "subject-run combinations\n\n")

# =============================================================================
# Process Each Subject-Run
# =============================================================================

all_start_epochs     <- list()
all_response_epochs  <- list()
all_baselines        <- list()
all_statistics       <- list()
all_start_timings    <- list()
all_response_timings <- list()

for (sub_indx in seq_along(data_list)) {
  sub_data <- data_list[[sub_indx]]
  run_id   <- as.integer(sub_data$Run[1])
  sub_id   <- as.integer(sub_data$Subject[1])

  cat("Processing Subject", sub_id, "Run", run_id, "...\n")

  if (sub_id %in% sheng2020_invalid_subs) {
    cat("  Skipping: invalid subject\n")
    next
  }

  sub_data <- fill_missing_samples_grid(
    sub_data,
    time_col      = "TimeSignal",
    event_col     = "StimulusName",
    sampling_rate = SHENG2020_SAMPLING_RATE,
    tolerance     = 0.5
  )

  data   <- convert_sheng2020_to_eyeris(sub_data, sampling_rate = SHENG2020_SAMPLING_RATE)
  result <- process_subject_run(
    data                   = data,
    sub_id                 = sub_id,
    run_id                 = run_id,
    dataset                = "sheng2020",
    apply_deblink          = FALSE,
    detransient_n          = DETRANSIENT_N,
    detransient_mad_thresh = DETRANSIENT_MAD_THRESH,
    lowpass_wp             = LOWPASS_WP,
    lowpass_ws             = LOWPASS_WS,
    lowpass_rp             = LOWPASS_RP,
    lowpass_rs             = LOWPASS_RS,
    output_sampling_rate   = OUTPUT_SAMPLING_RATE,
    apply_detrend          = TRUE,
    event_start_pattern    = SHENG2020_EVENT_START_PATTERN,
    event_response_pattern = SHENG2020_EVENT_RESPONSE_PATTERN,
    start_epoch_limits     = START_EPOCH_LIMITS,
    response_epoch_limits  = RESPONSE_EPOCH_LIMITS,
    baseline_period        = BASELINE_PERIOD,
    stats_epoch_limits     = STATS_EPOCH_LIMITS,
    trials_per_run         = SHENG2020_TRIALS_PER_RUN,
    timeseries_output_dir  = file.path(sheng2020_output_dir, "timeseries"),
    max_na_percentage      = MAX_NA_PERCENTAGE
  )

  if (is.null(result)) next

  all_start_epochs[[length(all_start_epochs) + 1]]         <- result$start_epoch
  if (!is.null(result$baseline)) {
    all_baselines[[length(all_baselines) + 1]]              <- result$baseline
  }
  all_response_epochs[[length(all_response_epochs) + 1]]   <- result$response_epoch
  all_statistics[[length(all_statistics) + 1]]             <- result$statistics
  all_start_timings[[length(all_start_timings) + 1]]       <- result$start_timings
  all_response_timings[[length(all_response_timings) + 1]] <- result$response_timings
}

# =============================================================================
# Combine and Save Outputs
# =============================================================================

cat("\n=== Saving Sheng2020 outputs ===\n")

start_epochs_df     <- bind_rows(all_start_epochs)
response_epochs_df  <- bind_rows(all_response_epochs)
baselines_df        <- bind_rows(all_baselines)
statistics_df       <- bind_rows(all_statistics)
start_timings_df    <- bind_rows(all_start_timings)
response_timings_df <- bind_rows(all_response_timings)

write.csv(start_timings_df,
          file = file.path(sheng2020_output_dir, "timeseries", "sheng2020_start_timings.csv"),
          row.names = FALSE)
write.csv(response_timings_df,
          file = file.path(sheng2020_output_dir, "timeseries", "sheng2020_response_timings.csv"),
          row.names = FALSE)

write.csv(start_epochs_df,
          file = file.path(sheng2020_output_dir, "sheng2020_start_aligned.csv"),
          row.names = FALSE)
cat("Saved start-aligned epochs:", nrow(start_epochs_df), "rows\n")

write.csv(response_epochs_df,
          file = file.path(sheng2020_output_dir, "sheng2020_response_aligned.csv"),
          row.names = FALSE)
cat("Saved response-aligned epochs:", nrow(response_epochs_df), "rows\n")

write.csv(baselines_df,
          file = file.path(sheng2020_output_dir, "sheng2020_baselines.csv"),
          row.names = FALSE)
cat("Saved baselines:", nrow(baselines_df), "rows\n")

write.csv(statistics_df,
          file = file.path(sheng2020_output_dir, "sheng2020_statistics.csv"),
          row.names = FALSE)
cat("Saved statistics:", nrow(statistics_df), "rows\n")

cat("\n=== Sheng2020 preprocessing complete ===\n")
