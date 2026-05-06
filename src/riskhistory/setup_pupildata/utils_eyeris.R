# =============================================================================
# Pupil Data Preprocessing Pipeline v2: Eyeris Utility Functions
# =============================================================================
# Shared functions for processing pupil data with the eyeris package.
# =============================================================================

library(eyeris)
library(dplyr)
library(stringr)
library(tibble)

# =============================================================================
# Core Processing Pipeline
# =============================================================================

#' Apply standard preprocessing pipeline to eyeris data
#'
#' @param data An eyeris object
#' @param apply_deblink Whether to apply deblink (TRUE for NARPS, FALSE for Sheng2020)
#' @param deblink_extend Extension for deblink in ms
#' @param detransient_n Number of samples for detransient
#' @param detransient_mad_thresh MAD threshold for detransient
#' @param lowpass_wp Passband frequency
#' @param lowpass_ws Stopband frequency
#' @param lowpass_rp Passband ripple
#' @param lowpass_rs Stopband attenuation
#' @param bins_per_second Output sampling rate
#' @param apply_detrend Whether to apply detrending
#' @return Preprocessed eyeris object
preprocess_eyeris <- function(
  data,
  apply_deblink = TRUE,
  deblink_extend = 100,
  detransient_n = 16,
  detransient_mad_thresh = NULL,
  lowpass_wp = 4,
  lowpass_ws = 8,
  lowpass_rp = 1,
  lowpass_rs = 35,
  bins_per_second = 8,
  apply_detrend = TRUE
) {
  # Deblink (only for datasets with raw blink data)
  if (apply_deblink) {
    data <- deblink(data, extend = deblink_extend)
  }

  # Detransient (remove artifacts)
  data <- detransient(data, n = detransient_n, mad_thresh = detransient_mad_thresh)

  # Interpolate missing values
  data <- interpolate(data, verbose = FALSE)

  # Low-pass filter
  data <- lpfilt(data, wp = lowpass_wp, ws = lowpass_ws, rp = lowpass_rp, rs = lowpass_rs)

  # Full time series before binning
  data_full <- detrend(data)
  data_full <- zscore(data_full)

  # Bin to 25Hz
  data_full$timeseries$block_1 <- dplyr::select(
    data_full$timeseries$block_1,
    block,
    time_orig,
    time_secs,
    eye_x,
    eye_y,
    all_of(data_full$latest$block_1)
  )
  data_full <- bin(data_full, bins_per_second = 25, method = "mean")

  # Full time series after filtering
  timeseries_time <- data_full$timeseries$block_1$time_secs
  pupil_filtered <- data_full$timeseries$block_1[[data_full$latest$block_1]]
  eye_x <- data_full$timeseries$block_1$eye_x
  eye_y <- data_full$timeseries$block_1$eye_y
  # If eye_x is Na, set to 0
  eye_x[is.na(eye_x)] <- 0
  eye_y[is.na(eye_y)] <- 0

  timeseries_pupil <- tibble::tibble(
    time_secs = timeseries_time,
    pupil_filtered = pupil_filtered,
    eye_x = eye_x,
    eye_y = eye_y
  )

  # Keep only necessary columns for efficiency
  current_col <- data$latest$block_1
  columns_selected <- c("block", "time_orig", "time_secs", current_col)
  data$timeseries$block_1 <- dplyr::select(
    data$timeseries$block_1,
    all_of(columns_selected)
  )

  # Detrend if requested
  if (apply_detrend) {
    data <- detrend(data)
  }

  # Z-score normalize
  data <- zscore(data)
  data$info$sample.rate <- bins_per_second

  # Bin the data
  data <- bin(data, bins_per_second = bins_per_second, method = "mean")

  # Rename the final column to a standard name
  final_col <- data$latest$block_1
  names(data$timeseries$block_1)[names(data$timeseries$block_1) == final_col] <- "pupil_processed"
  data$latest$block_1 <- "pupil_processed"

  # Return the preprocessed data and the full time series before binning
  results <- list(
    data = data,
    timeseries_pupil = timeseries_pupil
  )

  return(results)
}

# =============================================================================
# Statistics Calculation (Before Full Preprocessing)
# =============================================================================

#' Calculate data quality statistics for trials
#'
#' Computes NA statistics for both start-aligned and response-aligned epochs:
#' 1. Percent NaN baseline (-0.5 to 0 sec from start)
#' 2. Percent NaN start-aligned (0 to 4 sec from start)
#' 3. Percent NaN response-aligned (-2 to 2 sec from response)
#' 4. Longest NaN section in secs start-aligned
#' 5. Longest NaN section in secs response-aligned
#'
#' @param data An eyeris object (after deblink/detransient but before interpolation)
#' @param start_event_pattern Pattern for trial start events
#' @param response_event_pattern Pattern for response events
#' @param start_epoch_limits Limits for start-aligned epoching (should include baseline)
#' @param response_epoch_limits Limits for response-aligned epoching
#' @param baseline_window Window for baseline statistics relative to start
#' @param start_task_window Window for start-aligned task statistics
#' @param sub_id Subject ID
#' @param run_id Run ID
#' @return Data frame with statistics per trial
calculate_trial_statistics <- function(
  data,
  start_event_pattern,
  response_event_pattern,
  start_epoch_limits = c(-0.5, 4),
  response_epoch_limits = c(-2, 2),
  baseline_window = c(0, 0.5),
  start_task_window = c(0.5, 4.5),
  response_task_window = c(0, 4),
  trials_per_run,
  sub_id,
  run_id
) {
  # Find the pupil column (after deblink/detransient)
  pupil_col <- data$latest$block_1

  # ---------------------------------------------------------------------------
  # Start-aligned epoch statistics
  # ---------------------------------------------------------------------------
  start_epoch <- epoch(data, events = start_event_pattern, limits = start_epoch_limits)

  # Get the epoch data frame name (eyeris uses dynamic naming)
  epoch_name <- names(start_epoch)[grepl("^epoch_", names(start_epoch))][1]
  start_trials <- start_epoch[[epoch_name]]$block_1

  # Extract trial number from matched_event
  start_trials <- start_trials %>%
    mutate(
      trial = str_extract(matched_event, "Trial\\d+") %>%
        str_remove_all("Trial") %>%
        as.integer()
    )

  sr <- start_trials$hz[1]

  # Calculate start-aligned statistics per trial
  start_stats <- start_trials %>%
    group_by(trial) %>%
    summarise(
      # Baseline window NA percentage (-0.5 to 0 sec)
      pct_nan_baseline = mean(is.na(.data[[pupil_col]][
        timebin >= baseline_window[1] & timebin <= baseline_window[2]
      ])) * 100,

      # Task window NA percentage (0 to 4 sec)
      pct_nan_trial_startAligned = mean(is.na(.data[[pupil_col]][
        timebin >= start_task_window[1] & timebin <= start_task_window[2]
      ])) * 100,

      # Longest continuous NA segment in seconds (start-aligned)
      longest_nan_sec_startAligned = {
        task_vals <- .data[[pupil_col]][
          timebin >= start_task_window[1] & timebin <= start_task_window[2]
        ]
        if (all(!is.na(task_vals)) || length(task_vals) == 0) {
          0
        } else {
          runs <- rle(is.na(task_vals))
          if (any(runs$values == TRUE, na.rm = TRUE)) {
            max(runs$lengths[runs$values == TRUE], na.rm = TRUE) / sr
          } else {
            0
          }
        }
      },
      .groups = "drop"
    )

  # ---------------------------------------------------------------------------
  # Response-aligned epoch statistics
  # ---------------------------------------------------------------------------
  response_epoch <- epoch(data, events = response_event_pattern, limits = response_epoch_limits)

  # Get the epoch data frame name
  epoch_name <- names(response_epoch)[grepl("^epoch_", names(response_epoch))][1]
  response_trials <- response_epoch[[epoch_name]]$block_1

  # Extract trial number from matched_event
  response_trials <- response_trials %>%
    mutate(
      trial = str_extract(matched_event, "Trial\\d+") %>%
        str_remove_all("Trial") %>%
        as.integer()
    )

  # Calculate response-aligned statistics per trial
  response_stats <- response_trials %>%
    group_by(trial) %>%
    summarise(
      # Response window NA percentage (-2 to 2 sec)
      pct_nan_trial_responseAligned = mean(is.na(.data[[pupil_col]][
        timebin >= response_task_window[1] & timebin <= response_task_window[2]
      ])) * 100,

      # Longest continuous NA segment in seconds (response-aligned)
      longest_nan_sec_responseAligned = {
        resp_vals <- .data[[pupil_col]][
          timebin >= response_task_window[1] & timebin <= response_task_window[2]
        ]
        if (all(!is.na(resp_vals)) || length(resp_vals) == 0) {
          0
        } else {
          runs <- rle(is.na(resp_vals))
          if (any(runs$values == TRUE, na.rm = TRUE)) {
            max(runs$lengths[runs$values == TRUE], na.rm = TRUE) / sr
          } else {
            0
          }
        }
      },
      .groups = "drop"
    )

  # ---------------------------------------------------------------------------
  # Merge start and response statistics
  # ---------------------------------------------------------------------------
  summary_df <- start_stats %>%
    left_join(response_stats, by = "trial") %>%
    mutate(
      sub = sub_id,
      run = run_id
    ) %>%
    select(
      sub, run, trial,
      pct_nan_baseline,
      pct_nan_trial_startAligned,
      pct_nan_trial_responseAligned,
      longest_nan_sec_startAligned,
      longest_nan_sec_responseAligned
    )
  summary_df$trial <- summary_df$trial + trials_per_run * (run_id - 1)

  return(summary_df)
}

# =============================================================================
# Epoch Extraction
# =============================================================================

#' Extract epoched data aligned to trial start
#'
#' @param data Preprocessed eyeris object
#' @param event_pattern Event pattern for trial start
#' @param limits Epoch limits (seconds)
#' @param baseline_period Baseline period for subtraction
#' @param sub_id Subject ID
#' @param run_id Run ID
#' @param trials_per_run Number of trials per run (for global trial numbering)
#' @param dataset Dataset name ("narps" or "sheng2020")
#' @return List with epoch data frame and baseline data frame
extract_start_aligned_epoch <- function(
  data,
  event_pattern,
  limits = c(0, 4),
  baseline_period = c(-0.5, 0),
  sub_id,
  run_id,
  trials_per_run,
  dataset
) {
  # Epoch with baseline
  start_epoch <- epoch(
    data,
    events = event_pattern,
    baseline = TRUE,
    baseline_events = event_pattern,
    baseline_period = baseline_period,
    limits = limits
  )

  # Get epoch data frame name
  epoch_name <- names(start_epoch)[grepl("^epoch_", names(start_epoch))][1]
  trials_epoch <- start_epoch[[epoch_name]]$block_1

  # Extract trial number
  trials_epoch <- trials_epoch %>%
    mutate(
      trial = str_extract(matched_event, "Trial\\d+") %>%
        str_remove_all("Trial") %>%
        as.integer()
    )

  # Convert to global trial number
  trials_epoch$trial <- trials_epoch$trial + trials_per_run * (run_id - 1)
  trials_epoch$sub <- sub_id
  trials_epoch$dataset <- dataset

  # Select and rename columns
  epoch_df <- trials_epoch %>%
    select(sub, trial, dataset, timebin, pupil_size = pupil_processed)

  # Extract baseline
  baseline_name <- names(start_epoch)[grepl("^baseline_", names(start_epoch))][1]
  if (!is.null(baseline_name)) {
    baseline_means <- start_epoch[[baseline_name]]$block_1$baseline_means_by_epoch

    baseline_df <- trials_epoch %>%
      distinct(trial) %>%
      arrange(trial) %>%
      mutate(
        sub = sub_id,
        dataset = dataset,
        baseline_pupil = baseline_means
      )
  } else {
    baseline_df <- NULL
  }

  return(list(epoch = epoch_df, baseline = baseline_df))
}

#' Extract epoched data aligned to response
#'
#' @param data Preprocessed eyeris object
#' @param event_pattern Event pattern for response
#' @param limits Epoch limits (seconds)
#' @param sub_id Subject ID
#' @param run_id Run ID
#' @param trials_per_run Number of trials per run
#' @param dataset Dataset name
#' @return Data frame with epoched data
extract_response_aligned_epoch <- function(
  data,
  event_pattern,
  limits = c(-2, 2),
  sub_id,
  run_id,
  trials_per_run,
  dataset
) {
  # Epoch
  response_epoch <- epoch(data,
                          events = event_pattern,
                          limits = limits)

  # Get epoch data frame name
  epoch_name <- names(response_epoch)[grepl("^epoch_", names(response_epoch))][1]
  trials_epoch <- response_epoch[[epoch_name]]$block_1

  # Extract trial number
  trials_epoch <- trials_epoch %>%
    mutate(
      trial = str_extract(matched_event, "Trial\\d+") %>%
        str_remove_all("Trial") %>%
        as.integer()
    )

  # Convert to global trial number
  trials_epoch$trial <- trials_epoch$trial + trials_per_run * (run_id - 1)
  trials_epoch$sub <- sub_id
  trials_epoch$dataset <- dataset

  # Select and rename columns
  epoch_df <- trials_epoch %>%
    select(sub, trial, dataset, timebin, pupil_size = pupil_processed)

  return(epoch_df)
}

# =============================================================================
# Data Quality Check
# =============================================================================

#' Check if data has too much missing data to process
#'
#' @param data An eyeris object
#' @param max_na_percentage Maximum allowed NA percentage
#' @return TRUE if data should be skipped, FALSE otherwise
should_skip_data <- function(data, max_na_percentage = 90) {
  na_percentage <- sum(is.na(data$timeseries$block_1$pupil_raw)) /
    nrow(data$timeseries$block_1) * 100
  return(na_percentage > max_na_percentage)
}

#' Get NA percentage for a dataset
#'
#' @param data An eyeris object
#' @return NA percentage
get_na_percentage <- function(data) {
  sum(is.na(data$timeseries$block_1$pupil_raw)) /
    nrow(data$timeseries$block_1) * 100
}


# =============================================================================
# Additional helpers
# =============================================================================
extract_trial_times <- function(event_list, pattern_start, pattern_response, time_offset = 0) {
  start_events <-  event_list %>%
    filter(str_starts(text, pattern_start)) %>%
    mutate(
      trial = str_extract(text, "Trial\\d+") %>%
        str_remove_all("Trial") %>%
        as.integer(),
      time_secs = (time - time_offset) / 1000  # Convert ms to seconds
    ) %>%
    select(trial, time_secs)

  response_events <- event_list %>%
    filter(str_starts(text, pattern_response)) %>%
    mutate(
      trial = str_extract(text, "Trial\\d+") %>%
        str_remove_all("Trial") %>%
        as.integer(),
      time_secs = (time - time_offset) / 1000  # Convert ms to seconds
    ) %>%
    select(trial, time_secs)

  results <- list(
    start_events = start_events,
    response_events = response_events
  )
  return(results)
}

# =============================================================================
# Per-Subject-Run Processing (shared pipeline for NARPS and Sheng2020)
# =============================================================================

#' Run the full preprocessing pipeline for one subject-run
#'
#' Encapsulates the shared loop body: quality check, stats, preprocessing,
#' timeseries save, timing extraction, and epoch extraction.
#'
#' @param data An eyeris object (already loaded and, for Sheng2020, converted)
#' @param sub_id Integer subject ID
#' @param run_id Integer run ID
#' @param dataset Dataset label ("narps" or "sheng2020")
#' @param apply_deblink Whether to apply deblink (TRUE for NARPS, FALSE for Sheng2020)
#' @param deblink_extend Deblink extension in ms
#' @param detransient_n Number of samples for detransient
#' @param detransient_mad_thresh MAD threshold for detransient (NULL = auto)
#' @param lowpass_wp,lowpass_ws,lowpass_rp,lowpass_rs Low-pass filter parameters
#' @param output_sampling_rate Final sampling rate in Hz
#' @param apply_detrend Whether to apply linear detrending
#' @param event_start_pattern Glob pattern for trial-start events
#' @param event_response_pattern Glob pattern for response events
#' @param start_epoch_limits c(start, end) in seconds for trial epoch extraction
#' @param response_epoch_limits c(start, end) in seconds for response epoch
#' @param baseline_period c(start, end) in seconds for baseline subtraction
#' @param stats_epoch_limits Wider epoch limits used only for quality statistics
#' @param trials_per_run Number of trials per run (for global trial numbering)
#' @param timeseries_output_dir Directory for per-subject timeseries CSV output
#' @param max_na_percentage Skip subject-run if overall NA exceeds this threshold
#' @return Named list with start_epoch, baseline, response_epoch, statistics,
#'   start_timings, response_timings — or NULL if the subject-run was skipped.
process_subject_run <- function(
  data,
  sub_id,
  run_id,
  dataset,
  apply_deblink = TRUE,
  deblink_extend = 100,
  detransient_n = 16,
  detransient_mad_thresh = NULL,
  lowpass_wp = 4,
  lowpass_ws = 8,
  lowpass_rp = 1,
  lowpass_rs = 35,
  output_sampling_rate = 8,
  apply_detrend = TRUE,
  event_start_pattern,
  event_response_pattern,
  start_epoch_limits = c(0, 4),
  response_epoch_limits = c(-2, 2),
  baseline_period = c(-0.5, 0),
  stats_epoch_limits = c(-0.5, 4.5),
  trials_per_run,
  timeseries_output_dir,
  max_na_percentage = 90
) {
  # ---------------------------------------------------------------------------
  # Data quality check
  # ---------------------------------------------------------------------------
  na_pct <- get_na_percentage(data)
  cat("  NA percentage:", round(na_pct, 1), "%\n")

  if (should_skip_data(data, max_na_percentage)) {
    cat("  Skipping: >", max_na_percentage, "% missing data\n")
    return(NULL)
  }

  # ---------------------------------------------------------------------------
  # Quality statistics (before interpolation)
  # ---------------------------------------------------------------------------
  data_for_stats <- data
  if (apply_deblink) {
    data_for_stats <- deblink(data_for_stats, extend = deblink_extend)
  }
  data_for_stats <- detransient(data_for_stats, n = detransient_n, mad_thresh = detransient_mad_thresh)

  stats_df <- calculate_trial_statistics(
    data                 = data_for_stats,
    start_event_pattern  = event_start_pattern,
    response_event_pattern = event_response_pattern,
    start_epoch_limits   = stats_epoch_limits,
    response_epoch_limits = response_epoch_limits,
    baseline_window      = baseline_period,
    start_task_window    = start_epoch_limits,
    trials_per_run       = trials_per_run,
    sub_id               = sub_id,
    run_id               = run_id
  )
  stats_df$dataset <- dataset

  # ---------------------------------------------------------------------------
  # Full preprocessing pipeline
  # ---------------------------------------------------------------------------
  results <- preprocess_eyeris(
    data,
    apply_deblink        = apply_deblink,
    deblink_extend       = deblink_extend,
    detransient_n        = detransient_n,
    detransient_mad_thresh = detransient_mad_thresh,
    lowpass_wp           = lowpass_wp,
    lowpass_ws           = lowpass_ws,
    lowpass_rp           = lowpass_rp,
    lowpass_rs           = lowpass_rs,
    bins_per_second      = output_sampling_rate,
    apply_detrend        = apply_detrend
  )
  data <- results$data

  # ---------------------------------------------------------------------------
  # Save full timeseries
  # ---------------------------------------------------------------------------
  ensure_dir(timeseries_output_dir)
  timeseries_path <- file.path(
    timeseries_output_dir,
    sprintf("sub-%03d_run-%d_timeseries.csv", sub_id, run_id)
  )
  write.csv(results$timeseries_pupil, file = timeseries_path, row.names = FALSE)
  cat("  Saved timeseries:", timeseries_path, "\n")

  # ---------------------------------------------------------------------------
  # Extract trial onset and response timings
  # ---------------------------------------------------------------------------
  start_time_run <- data$timeseries$block_1$time_orig[1]
  timings <- extract_trial_times(
    data$events$block_1,
    event_start_pattern,
    event_response_pattern,
    time_offset = start_time_run
  )

  start_timings           <- timings$start_events
  start_timings$run_id    <- run_id
  start_timings$sub_id    <- sub_id

  response_timings        <- timings$response_events
  response_timings$run_id <- run_id
  response_timings$sub_id <- sub_id

  # ---------------------------------------------------------------------------
  # Extract start-aligned epochs
  # ---------------------------------------------------------------------------
  start_result <- extract_start_aligned_epoch(
    data           = data,
    event_pattern  = event_start_pattern,
    limits         = start_epoch_limits,
    baseline_period = baseline_period,
    sub_id         = sub_id,
    run_id         = run_id,
    trials_per_run = trials_per_run,
    dataset        = dataset
  )

  # ---------------------------------------------------------------------------
  # Extract response-aligned epochs
  # ---------------------------------------------------------------------------
  response_df <- extract_response_aligned_epoch(
    data           = data,
    event_pattern  = event_response_pattern,
    limits         = response_epoch_limits,
    sub_id         = sub_id,
    run_id         = run_id,
    trials_per_run = trials_per_run,
    dataset        = dataset
  )

  cat("  Done\n")

  list(
    start_epoch     = start_result$epoch,
    baseline        = start_result$baseline,
    response_epoch  = response_df,
    statistics      = stats_df,
    start_timings   = start_timings,
    response_timings = response_timings
  )
}