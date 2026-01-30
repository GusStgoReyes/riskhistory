# =============================================================================
# Pupil Data Preprocessing Pipeline v2: Sheng2020 Dataset
# =============================================================================
# Preprocesses Sheng2020 pupil data (CSV format) and saves intermediate outputs.
# This dataset requires conversion to eyeris format before processing.
# =============================================================================

# Load dependencies
library(eyeris)
library(dplyr)
library(stringr)

# Source config and utilities
script_dir <- dirname(sys.frame(1)$ofile)
source(file.path(script_dir, "0_config.R"))
source(file.path(script_dir, "utils_eyeris.R"))
source(file.path(script_dir, "utils_sheng2020.R"))

# =============================================================================
# Setup
# =============================================================================

cat("=== Sheng2020 Pupil Data Preprocessing ===\n")
cat("Input file:", sheng2020_input_file, "\n")
cat("Output directory:", sheng2020_output_dir, "\n\n")

# Ensure output directory exists
ensure_dir(sheng2020_output_dir)

# Load and clean data
cat("Loading Sheng2020 data...\n")
all_data <- load_sheng2020_data(sheng2020_input_file)
cat("Loaded", nrow(all_data), "rows\n")

# Split by subject and run
data_list <- split_sheng2020_by_subject_run(all_data)
cat("Found", length(data_list), "subject-run combinations\n\n")

# =============================================================================
# Process Each Subject-Run
# =============================================================================

# Storage for combined results
all_start_epochs <- list()
all_response_epochs <- list()
all_baselines <- list()
all_statistics <- list()
all_start_timings <- list()
all_response_timings <- list()

for (sub_indx in seq_along(data_list)) {
  sub_data <- data_list[[sub_indx]]
  run_id <- as.integer(sub_data$Run[1])
  sub_id <- as.integer(sub_data$Subject[1])

  cat("Processing Subject", sub_id, "Run", run_id, "...\n")

  # Skip if in invalid subjects list
  if (sub_id %in% sheng2020_invalid_subs) {
    cat("  Skipping: invalid subject\n")
    next
  }
  # Fill missing time samples
  sub_data <- fill_missing_samples_grid(
    sub_data,
    time_col = "TimeSignal",
    event_col = "StimulusName",
    sampling_rate = SHENG2020_SAMPLING_RATE,
    tolerance = 0.5
  )

  # Convert to eyeris format
  data <- convert_sheng2020_to_eyeris(sub_data, sampling_rate = SHENG2020_SAMPLING_RATE)

  # Check data quality
  na_pct <- get_na_percentage(data)
  cat("  NA percentage:", round(na_pct, 1), "%\n")

  if (should_skip_data(data, MAX_NA_PERCENTAGE)) {
    cat("  Skipping: >", MAX_NA_PERCENTAGE, "% missing data\n")
    next
  }

  # ---------------------------------------------------------------------------
  # Calculate statistics BEFORE full preprocessing (after detransient only)
  # Note: Sheng2020 doesn't need deblink (already preprocessed for blinks)
  # ---------------------------------------------------------------------------
  data_for_stats <- detransient(data, n = DETRANSIENT_N, mad_thresh = DETRANSIENT_MAD_THRESH)

  stats_df <- calculate_trial_statistics(
    data = data_for_stats,
    start_event_pattern = SHENG2020_EVENT_START_PATTERN,
    response_event_pattern = SHENG2020_EVENT_RESPONSE_PATTERN,
    start_epoch_limits = STATS_EPOCH_LIMITS,
    response_epoch_limits = RESPONSE_EPOCH_LIMITS,
    baseline_window = BASELINE_PERIOD,
    start_task_window = START_EPOCH_LIMITS,
    trials_per_run = SHENG2020_TRIALS_PER_RUN,
    sub_id = sub_id,
    run_id = run_id
  )
  stats_df$dataset <- "sheng2020"
  all_statistics[[length(all_statistics) + 1]] <- stats_df

  # ---------------------------------------------------------------------------
  # Full preprocessing pipeline
  # Note: No deblink for Sheng2020, and no detrend (following original pipeline)
  # ---------------------------------------------------------------------------
  results <- preprocess_eyeris(
    data,
    apply_deblink = FALSE,  # Sheng2020 already has blink correction
    detransient_n = DETRANSIENT_N,
    detransient_mad_thresh = DETRANSIENT_MAD_THRESH,
    lowpass_wp = LOWPASS_WP,
    lowpass_ws = LOWPASS_WS,
    lowpass_rp = LOWPASS_RP,
    lowpass_rs = LOWPASS_RS,
    bins_per_second = OUTPUT_SAMPLING_RATE,
    apply_detrend = TRUE
  )
  data <- results$data

  # Full time series after filtering (not binned)
  timeseries_pupil <- results$timeseries_pupil
  # Save the timeseries output
  timeseries_output_path <- file.path(
    sheng2020_output_dir,
    "timeseries",
    sprintf("sub-%03d_run-%d_timeseries.csv", sub_id, run_id)
  )
  write.csv(
    timeseries_pupil,
    file = timeseries_output_path,
    row.names = FALSE
  )
  cat("  Saved timeseries to", timeseries_output_path, "\n")
  start_time_run <- data$timeseries$block_1$time_orig[1]
  # Collect the timing of the Stimulus onset
  timings <- extract_trial_times(data$events$block_1,
                                 SHENG2020_EVENT_START_PATTERN,
                                 SHENG2020_EVENT_RESPONSE_PATTERN,
                                 time_offset = start_time_run)
  start_timings <- timings$start_events
  start_timings$run_id <- run_id
  start_timings$sub_id <- sub_id
  all_start_timings[[length(all_start_timings) + 1]] <- start_timings
  print(str(head(start_timings)))

  response_timings <- timings$response_events
  response_timings$run_id <- run_id
  response_timings$sub_id <- sub_id
  all_response_timings[[length(all_response_timings) + 1]] <- response_timings
  print(str(head(response_timings)))

  # ---------------------------------------------------------------------------
  # Extract start-aligned epochs
  # ---------------------------------------------------------------------------
  start_result <- extract_start_aligned_epoch(
    data = data,
    event_pattern = SHENG2020_EVENT_START_PATTERN,
    limits = START_EPOCH_LIMITS,
    baseline_period = BASELINE_PERIOD,
    sub_id = sub_id,
    run_id = run_id,
    trials_per_run = SHENG2020_TRIALS_PER_RUN,
    dataset = "sheng2020"
  )

  all_start_epochs[[length(all_start_epochs) + 1]] <- start_result$epoch
  if (!is.null(start_result$baseline)) {
    all_baselines[[length(all_baselines) + 1]] <- start_result$baseline
  }

  # ---------------------------------------------------------------------------
  # Extract response-aligned epochs
  # ---------------------------------------------------------------------------
  response_df <- extract_response_aligned_epoch(
    data = data,
    event_pattern = SHENG2020_EVENT_RESPONSE_PATTERN,
    limits = RESPONSE_EPOCH_LIMITS,
    sub_id = sub_id,
    run_id = run_id,
    trials_per_run = SHENG2020_TRIALS_PER_RUN,
    dataset = "sheng2020"
  )

  all_response_epochs[[length(all_response_epochs) + 1]] <- response_df

  cat("  Done\n")
}

# =============================================================================
# Combine and Save Outputs
# =============================================================================

cat("\n=== Saving Sheng2020 outputs ===\n")

# Combine all data frames
start_epochs_df <- bind_rows(all_start_epochs)
response_epochs_df <- bind_rows(all_response_epochs)
baselines_df <- bind_rows(all_baselines)
statistics_df <- bind_rows(all_statistics)
start_timings_df <- bind_rows(all_start_timings)
response_timings_df <- bind_rows(all_response_timings)

# Save to CSV
write.csv(
  start_timings_df,
  file = file.path(sheng2020_output_dir, "timeseries", "sheng2020_start_timings.csv"),
  row.names = FALSE
)

write.csv(
  response_timings_df,
  file = file.path(sheng2020_output_dir, "timeseries", "sheng2020_response_timings.csv"),
  row.names = FALSE
)

write.csv(
  start_epochs_df,
  file = file.path(sheng2020_output_dir, "sheng2020_start_aligned.csv"),
  row.names = FALSE
)
cat("Saved start-aligned epochs:", nrow(start_epochs_df), "rows\n")

write.csv(
  response_epochs_df,
  file = file.path(sheng2020_output_dir, "sheng2020_response_aligned.csv"),
  row.names = FALSE
)
cat("Saved response-aligned epochs:", nrow(response_epochs_df), "rows\n")

write.csv(
  baselines_df,
  file = file.path(sheng2020_output_dir, "sheng2020_baselines.csv"),
  row.names = FALSE
)
cat("Saved baselines:", nrow(baselines_df), "rows\n")

write.csv(
  statistics_df,
  file = file.path(sheng2020_output_dir, "sheng2020_statistics.csv"),
  row.names = FALSE
)
cat("Saved statistics:", nrow(statistics_df), "rows\n")

cat("\n=== Sheng2020 preprocessing complete ===\n")
