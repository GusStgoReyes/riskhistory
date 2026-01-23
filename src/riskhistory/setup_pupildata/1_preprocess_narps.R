# =============================================================================
# Pupil Data Preprocessing Pipeline v2: NARPS Dataset
# =============================================================================
# Preprocesses NARPS pupil data (.asc files) and saves intermediate outputs.
# =============================================================================

# Load dependencies
library(eyeris)
library(dplyr)
library(stringr)

# Source config and utilities
script_dir <- dirname(sys.frame(1)$ofile)
source(file.path(script_dir, "0_config.R"))
source(file.path(script_dir, "utils_eyeris.R"))

# =============================================================================
# Setup
# =============================================================================

cat("=== NARPS Pupil Data Preprocessing ===\n")
cat("Input directory:", narps_input_dir, "\n")
cat("Output directory:", narps_output_dir, "\n\n")

# Ensure output directory exists
ensure_dir(narps_output_dir)

# Get list of ASC files
asc_files <- list.files(narps_input_dir, pattern = "\\.asc$", full.names = TRUE)
cat("Found", length(asc_files), "ASC files\n\n")

if (length(asc_files) == 0) {
  stop("No ASC files found in ", narps_input_dir)
}

# =============================================================================
# Process Each File
# =============================================================================

# Storage for combined results
all_start_epochs <- list()
all_response_epochs <- list()
all_baselines <- list()
all_statistics <- list()

for (file in asc_files) {
  # Extract subject and run IDs from filename
  subject_id <- stringr::str_extract(file, "sub-\\d{3}")
  sub_id <- str_remove(subject_id, "sub-") %>% as.integer()
  run_id <- stringr::str_extract(file, "run-\\d") %>%
    str_remove("run-") %>%
    as.integer()

  cat("Processing", subject_id, "run", run_id, "...\n")

  # Skip if not in valid subjects list
  if (subject_id %in% narps_invalid_subs) {
    cat("  Skipping: invalid subject \n")
    next
  }

  # Load the data
  data <- load_asc(file, block = "auto")

  # Check data quality
  na_pct <- get_na_percentage(data)
  cat("  NA percentage:", round(na_pct, 1), "%\n")

  if (should_skip_data(data, MAX_NA_PERCENTAGE)) {
    cat("  Skipping: >", MAX_NA_PERCENTAGE, "% missing data\n")
    next
  }

  # ---------------------------------------------------------------------------
  # Calculate statistics BEFORE full preprocessing (after deblink/detransient)
  # ---------------------------------------------------------------------------
  data_for_stats <- deblink(data, extend = DEBLINK_EXTEND)
  data_for_stats <- detransient(data_for_stats, n = DETRANSIENT_N, mad_thresh = DETRANSIENT_MAD_THRESH)

  stats_df <- calculate_trial_statistics(
    data = data_for_stats,
    start_event_pattern = NARPS_EVENT_START_PATTERN,
    response_event_pattern = NARPS_EVENT_RESPONSE_PATTERN,
    start_epoch_limits = STATS_EPOCH_LIMITS,
    response_epoch_limits = RESPONSE_EPOCH_LIMITS,
    baseline_window = BASELINE_PERIOD,
    start_task_window = START_EPOCH_LIMITS,
    sub_id = sub_id,
    run_id = run_id
  )
  stats_df$dataset <- "narps"
  all_statistics[[length(all_statistics) + 1]] <- stats_df

  # ---------------------------------------------------------------------------
  # Full preprocessing pipeline
  # ---------------------------------------------------------------------------
  data <- preprocess_eyeris(
    data,
    apply_deblink = TRUE,
    deblink_extend = DEBLINK_EXTEND,
    detransient_n = DETRANSIENT_N,
    detransient_mad_thresh = DETRANSIENT_MAD_THRESH,
    lowpass_wp = LOWPASS_WP,
    lowpass_ws = LOWPASS_WS,
    lowpass_rp = LOWPASS_RP,
    lowpass_rs = LOWPASS_RS,
    bins_per_second = OUTPUT_SAMPLING_RATE,
    apply_detrend = TRUE
  )

  # ---------------------------------------------------------------------------
  # Extract start-aligned epochs
  # ---------------------------------------------------------------------------
  start_result <- extract_start_aligned_epoch(
    data = data,
    event_pattern = NARPS_EVENT_START_PATTERN,
    limits = START_EPOCH_LIMITS,
    baseline_period = BASELINE_PERIOD,
    sub_id = sub_id,
    run_id = run_id,
    trials_per_run = NARPS_TRIALS_PER_RUN,
    dataset = "narps"
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
    event_pattern = NARPS_EVENT_RESPONSE_PATTERN,
    limits = RESPONSE_EPOCH_LIMITS,
    sub_id = sub_id,
    run_id = run_id,
    trials_per_run = NARPS_TRIALS_PER_RUN,
    dataset = "narps"
  )

  all_response_epochs[[length(all_response_epochs) + 1]] <- response_df

  cat("  Done\n")
}

# =============================================================================
# Combine and Save Outputs
# =============================================================================

cat("\n=== Saving NARPS outputs ===\n")

# Combine all data frames
start_epochs_df <- bind_rows(all_start_epochs)
response_epochs_df <- bind_rows(all_response_epochs)
baselines_df <- bind_rows(all_baselines)
statistics_df <- bind_rows(all_statistics)

# Save to CSV
write.csv(
  start_epochs_df,
  file = file.path(narps_output_dir, "narps_start_aligned.csv"),
  row.names = FALSE
)
cat("Saved start-aligned epochs:", nrow(start_epochs_df), "rows\n")

write.csv(
  response_epochs_df,
  file = file.path(narps_output_dir, "narps_response_aligned.csv"),
  row.names = FALSE
)
cat("Saved response-aligned epochs:", nrow(response_epochs_df), "rows\n")

write.csv(
  baselines_df,
  file = file.path(narps_output_dir, "narps_baselines.csv"),
  row.names = FALSE
)
cat("Saved baselines:", nrow(baselines_df), "rows\n")

write.csv(
  statistics_df,
  file = file.path(narps_output_dir, "narps_statistics.csv"),
  row.names = FALSE
)
cat("Saved statistics:", nrow(statistics_df), "rows\n")

cat("\n=== NARPS preprocessing complete ===\n")
