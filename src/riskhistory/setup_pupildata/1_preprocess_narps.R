# =============================================================================
# Pupil Data Preprocessing Pipeline: NARPS Dataset
# =============================================================================
# Preprocesses NARPS pupil data (.asc files) and saves intermediate outputs.
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

# =============================================================================
# Setup
# =============================================================================

cat("=== NARPS Pupil Data Preprocessing ===\n")
cat("Input directory:", narps_input_dir, "\n")
cat("Output directory:", narps_output_dir, "\n\n")

ensure_dir(narps_output_dir)

asc_files <- list.files(narps_input_dir, pattern = "\\.asc$", full.names = TRUE)
cat("Found", length(asc_files), "ASC files\n\n")

if (length(asc_files) == 0) {
  stop("No ASC files found in ", narps_input_dir)
}

# =============================================================================
# Process Each File
# =============================================================================

all_start_epochs    <- list()
all_response_epochs <- list()
all_baselines       <- list()
all_statistics      <- list()
all_start_timings   <- list()
all_response_timings <- list()

for (file in asc_files) {
  subject_id <- stringr::str_extract(file, "sub-\\d{3}")
  sub_id     <- str_remove(subject_id, "sub-") %>% as.integer()
  run_id     <- stringr::str_extract(file, "run-\\d") %>%
    str_remove("run-") %>%
    as.integer()

  cat("Processing", subject_id, "run", run_id, "...\n")

  if (subject_id %in% narps_invalid_subs) {
    cat("  Skipping: invalid subject\n")
    next
  }

  data   <- load_asc(file, block = "auto")
  result <- process_subject_run(
    data                   = data,
    sub_id                 = sub_id,
    run_id                 = run_id,
    dataset                = "narps",
    apply_deblink          = TRUE,
    deblink_extend         = DEBLINK_EXTEND,
    detransient_n          = DETRANSIENT_N,
    detransient_mad_thresh = DETRANSIENT_MAD_THRESH,
    lowpass_wp             = LOWPASS_WP,
    lowpass_ws             = LOWPASS_WS,
    lowpass_rp             = LOWPASS_RP,
    lowpass_rs             = LOWPASS_RS,
    output_sampling_rate   = OUTPUT_SAMPLING_RATE,
    apply_detrend          = TRUE,
    event_start_pattern    = NARPS_EVENT_START_PATTERN,
    event_response_pattern = NARPS_EVENT_RESPONSE_PATTERN,
    start_epoch_limits     = START_EPOCH_LIMITS,
    response_epoch_limits  = RESPONSE_EPOCH_LIMITS,
    baseline_period        = BASELINE_PERIOD,
    stats_epoch_limits     = STATS_EPOCH_LIMITS,
    trials_per_run         = NARPS_TRIALS_PER_RUN,
    timeseries_output_dir  = file.path(narps_output_dir, "timeseries"),
    max_na_percentage      = MAX_NA_PERCENTAGE
  )

  if (is.null(result)) next

  all_start_epochs[[length(all_start_epochs) + 1]]       <- result$start_epoch
  if (!is.null(result$baseline)) {
    all_baselines[[length(all_baselines) + 1]]            <- result$baseline
  }
  all_response_epochs[[length(all_response_epochs) + 1]] <- result$response_epoch
  all_statistics[[length(all_statistics) + 1]]           <- result$statistics
  all_start_timings[[length(all_start_timings) + 1]]     <- result$start_timings
  all_response_timings[[length(all_response_timings) + 1]] <- result$response_timings
}

# =============================================================================
# Combine and Save Outputs
# =============================================================================

cat("\n=== Saving NARPS outputs ===\n")

start_epochs_df    <- bind_rows(all_start_epochs)
response_epochs_df <- bind_rows(all_response_epochs)
baselines_df       <- bind_rows(all_baselines)
statistics_df      <- bind_rows(all_statistics)
start_timings_df   <- bind_rows(all_start_timings)
response_timings_df <- bind_rows(all_response_timings)

write.csv(start_timings_df,
          file = file.path(narps_output_dir, "timeseries", "narps_start_timings.csv"),
          row.names = FALSE)
write.csv(response_timings_df,
          file = file.path(narps_output_dir, "timeseries", "narps_response_timings.csv"),
          row.names = FALSE)

write.csv(start_epochs_df,
          file = file.path(narps_output_dir, "narps_start_aligned.csv"),
          row.names = FALSE)
cat("Saved start-aligned epochs:", nrow(start_epochs_df), "rows\n")

write.csv(response_epochs_df,
          file = file.path(narps_output_dir, "narps_response_aligned.csv"),
          row.names = FALSE)
cat("Saved response-aligned epochs:", nrow(response_epochs_df), "rows\n")

write.csv(baselines_df,
          file = file.path(narps_output_dir, "narps_baselines.csv"),
          row.names = FALSE)
cat("Saved baselines:", nrow(baselines_df), "rows\n")

write.csv(statistics_df,
          file = file.path(narps_output_dir, "narps_statistics.csv"),
          row.names = FALSE)
cat("Saved statistics:", nrow(statistics_df), "rows\n")

cat("\n=== NARPS preprocessing complete ===\n")
