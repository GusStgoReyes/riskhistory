# =============================================================================
# Pupil Data Preprocessing Pipeline: Combine Datasets
# =============================================================================
# Combines NARPS and Sheng2020 preprocessed pupil data into final output files:
# - pupil_data_alignedStartTime.csv
# - pupil_data_alignedResponseTime.csv
# - pupil_data_statistics.csv
# =============================================================================

# Load dependencies
library(dplyr)
library(tidyr)

# Source config
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

# =============================================================================
# Setup
# =============================================================================

cat("=== Combining Pupil Datasets ===\n\n")

# Ensure output directory exists
ensure_dir(pupil_output_dir)

# NARPS assigns condition by subject parity (odd = equalIndifference, even = equalRange)
add_narps_dataset <- function(df) {
  if (is.null(df)) return(NULL)
  df %>% mutate(dataset = ifelse(sub %% 2 == 1, "equalIndifference", "equalRange"))
}

# =============================================================================
# Load Intermediate Files
# =============================================================================

cat("Loading NARPS data...\n")
narps_start <- tryCatch(
  read.csv(file.path(narps_output_dir, "narps_start_aligned.csv")),
  error = function(e) {
    cat("  Warning: Could not load NARPS start-aligned data\n")
    NULL
  }
)
narps_response <- tryCatch(
  read.csv(file.path(narps_output_dir, "narps_response_aligned.csv")),
  error = function(e) {
    cat("  Warning: Could not load NARPS response-aligned data\n")
    NULL
  }
)
narps_baselines <- tryCatch(
  read.csv(file.path(narps_output_dir, "narps_baselines.csv")),
  error = function(e) {
    cat("  Warning: Could not load NARPS baselines\n")
    NULL
  }
)
narps_stats <- tryCatch(
  read.csv(file.path(narps_output_dir, "narps_statistics.csv")),
  error = function(e) {
    cat("  Warning: Could not load NARPS statistics\n")
    NULL
  }
)

cat("Loading Sheng2020 data...\n")
sheng_start <- tryCatch(
  read.csv(file.path(sheng2020_output_dir, "sheng2020_start_aligned.csv")),
  error = function(e) {
    cat("  Warning: Could not load Sheng2020 start-aligned data\n")
    NULL
  }
)
sheng_response <- tryCatch(
  read.csv(file.path(sheng2020_output_dir, "sheng2020_response_aligned.csv")),
  error = function(e) {
    cat("  Warning: Could not load Sheng2020 response-aligned data\n")
    NULL
  }
)
sheng_baselines <- tryCatch(
  read.csv(file.path(sheng2020_output_dir, "sheng2020_baselines.csv")),
  error = function(e) {
    cat("  Warning: Could not load Sheng2020 baselines\n")
    NULL
  }
)
sheng_stats <- tryCatch(
  read.csv(file.path(sheng2020_output_dir, "sheng2020_statistics.csv")),
  error = function(e) {
    cat("  Warning: Could not load Sheng2020 statistics\n")
    NULL
  }
)

# =============================================================================
# Combine Start-Aligned Data
# =============================================================================

cat("\nCombining start-aligned data...\n")

# Combine epoch data
narps_start     <- add_narps_dataset(narps_start)
start_epochs    <- bind_rows(narps_start, sheng_start)

# Combine baselines
narps_baselines <- add_narps_dataset(narps_baselines)
baselines <- bind_rows(
  narps_baselines,
  sheng_baselines
)

# Merge baselines into epochs
if (nrow(start_epochs) > 0 && nrow(baselines) > 0) {
  start_aligned <- start_epochs %>%
    left_join(
      baselines %>% select(sub, trial, dataset, baseline_pupil),
      by = c("sub", "trial", "dataset")
    ) %>%
    select(sub, trial, dataset, timebin, pupil_size, baseline_pupil)
} else {
  start_aligned <- start_epochs %>%
    mutate(baseline_pupil = NA_real_) %>%
    select(sub, trial, dataset, timebin, pupil_size, baseline_pupil)
  cat("  Warning: Missing baseline data, using NA\n")
}

# =============================================================================
# Combine Response-Aligned Data
# =============================================================================

cat("\nCombining response-aligned data...\n")

narps_response   <- add_narps_dataset(narps_response)
response_aligned <- bind_rows(narps_response, sheng_response)

# Make the timebin be relative to response time
response_aligned <- response_aligned %>%
  mutate(timebin = timebin + RESPONSE_EPOCH_LIMITS[1])

# Add baseline from start-aligned data (baseline is computed at trial start)
if (nrow(response_aligned) > 0 && nrow(baselines) > 0) {
  response_aligned <- response_aligned %>%
    left_join(
      baselines %>% select(sub, trial, dataset, baseline_pupil),
      by = c("sub", "trial", "dataset")
    ) %>%
    select(sub, trial, dataset, timebin, pupil_size, baseline_pupil)
} else {
  response_aligned <- response_aligned %>%
    mutate(baseline_pupil = NA_real_) %>%
    select(sub, trial, dataset, timebin, pupil_size, baseline_pupil)
}

# =============================================================================
# Combine Statistics
# =============================================================================

cat("\nCombining statistics...\n")

narps_stats    <- add_narps_dataset(narps_stats)
stats_combined <- bind_rows(narps_stats, sheng_stats)

if (nrow(stats_combined) > 0) {
  # Statistics are now computed properly in calculate_trial_statistics()
  # with separate epochs for start-aligned and response-aligned windows
  statistics <- stats_combined %>%
    select(
      sub, trial, dataset, run,
      pct_nan_baseline,
      pct_nan_trial_startAligned,
      pct_nan_trial_responseAligned,
      longest_nan_sec_startAligned,
      longest_nan_sec_responseAligned
    )
} else {
  statistics <- data.frame()
  cat("  Warning: No statistics data available\n")
}

# =============================================================================
# Save Final Outputs
# =============================================================================

cat("\n=== Saving final outputs ===\n")

write.csv(
  start_aligned,
  file = output_start_aligned,
  row.names = FALSE
)
cat("Saved:", output_start_aligned, "\n")
cat("  Columns:", paste(names(start_aligned), collapse = ", "), "\n")

write.csv(
  response_aligned,
  file = output_response_aligned,
  row.names = FALSE
)
cat("Saved:", output_response_aligned, "\n")
cat("  Columns:", paste(names(response_aligned), collapse = ", "), "\n")

write.csv(
  statistics,
  file = output_statistics,
  row.names = FALSE
)
cat("Saved:", output_statistics, "\n")
cat("  Columns:", paste(names(statistics), collapse = ", "), "\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n=== Summary ===\n")
if (nrow(start_aligned) > 0) {
  cat("  equalIndifference subjects:", n_distinct(start_aligned$sub[start_aligned$dataset == "equalIndifference"]), "\n")
  cat("  equalRange subjects:", n_distinct(start_aligned$sub[start_aligned$dataset == "equalRange"]), "\n")
  cat("  Sheng2020 subjects:", n_distinct(start_aligned$sub[start_aligned$dataset == "sheng2020"]), "\n")
  cat("  Timebin range:", min(start_aligned$timebin), "to", max(start_aligned$timebin), "sec\n")
}
cat("\n=== Dataset combination complete ===\n")
