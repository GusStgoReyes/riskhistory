# =============================================================================
# Check Trial Statistics: Mean trials per subject per timebin per dataset
# =============================================================================
# This script computes and saves trial inclusion statistics after applying
# behavioral and pupil exclusion criteria.
# =============================================================================

# =============================================================================
# Compute trial statistics
# =============================================================================
message("Computing trial statistics...")

# Start-aligned: mean trials per subject per timebin per dataset
stats_start <- data_start_clean %>%
  group_by(dataset, timebin, sub) %>%
  summarize(n_trials = n(), .groups = "drop") %>%
  group_by(dataset, timebin) %>%
  summarize(
    mean_trials = mean(n_trials),
    sd_trials = sd(n_trials),
    min_trials = min(n_trials),
    max_trials = max(n_trials),
    n_subjects = n(),
    .groups = "drop"
  )

# Response-aligned: mean trials per subject per timebin per dataset
stats_response <- data_response_clean %>%
  group_by(dataset, timebin, sub) %>%
  summarize(n_trials = n(), .groups = "drop") %>%
  group_by(dataset, timebin) %>%
  summarize(
    mean_trials = mean(n_trials),
    sd_trials = sd(n_trials),
    min_trials = min(n_trials),
    max_trials = max(n_trials),
    n_subjects = n(),
    .groups = "drop"
  )

# Overall summary (collapsed across timebins)
overall_start <- data_start_clean %>%
  group_by(dataset, sub) %>%
  summarize(n_trials = n_distinct(trial), .groups = "drop") %>%
  group_by(dataset) %>%
  summarize(
    mean_trials_per_subject = mean(n_trials),
    sd_trials = sd(n_trials),
    min_trials = min(n_trials),
    max_trials = max(n_trials),
    n_subjects = n(),
    total_trials = sum(n_trials),
    .groups = "drop"
  )

overall_response <- data_response_clean %>%
  group_by(dataset, sub) %>%
  summarize(n_trials = n_distinct(trial), .groups = "drop") %>%
  group_by(dataset) %>%
  summarize(
    mean_trials_per_subject = mean(n_trials),
    sd_trials = sd(n_trials),
    min_trials = min(n_trials),
    max_trials = max(n_trials),
    n_subjects = n(),
    total_trials = sum(n_trials),
    .groups = "drop"
  )

# =============================================================================
# Write results to text file
# =============================================================================
output_file <- file.path(analysis_results_path, "1_trial_statistics.txt")

sink(output_file)

cat("=============================================================================\n")
cat("Trial Statistics for Pupil Analysis\n")
cat("=============================================================================\n\n")

cat("OVERALL SUMMARY (unique trials per subject)\n")
cat("-----------------------------------------------------------------------------\n\n")

cat("Start-aligned data:\n")
print(as.data.frame(overall_start), row.names = FALSE)
cat("\n")

cat("Response-aligned data:\n")
print(as.data.frame(overall_response), row.names = FALSE)
cat("\n")

cat("\n=============================================================================\n")
cat("MEAN TRIALS PER SUBJECT PER TIMEBIN PER DATASET\n")
cat("=============================================================================\n\n")

cat("Start-aligned data:\n")
cat("-----------------------------------------------------------------------------\n")
for (ds in datasets) {
  cat(sprintf("\nDataset: %s\n", ds))
  ds_stats <- stats_start %>% filter(dataset == ds)
  if (nrow(ds_stats) > 0) {
    cat(sprintf("  Timebins: %d (range: %.2f to %.2f)\n",
                nrow(ds_stats), min(ds_stats$timebin), max(ds_stats$timebin)))
    cat(sprintf("  Mean trials per subject per timebin: %.2f (SD = %.2f)\n",
                mean(ds_stats$mean_trials), mean(ds_stats$sd_trials)))
    cat(sprintf("  Range: %.0f to %.0f\n",
                min(ds_stats$min_trials), max(ds_stats$max_trials)))
  } else {
    cat("  No data available\n")
  }
}

cat("\n\nResponse-aligned data:\n")
cat("-----------------------------------------------------------------------------\n")
for (ds in datasets) {
  cat(sprintf("\nDataset: %s\n", ds))
  ds_stats <- stats_response %>% filter(dataset == ds)
  if (nrow(ds_stats) > 0) {
    cat(sprintf("  Timebins: %d (range: %.2f to %.2f)\n",
                nrow(ds_stats), min(ds_stats$timebin), max(ds_stats$timebin)))
    cat(sprintf("  Mean trials per subject per timebin: %.2f (SD = %.2f)\n",
                mean(ds_stats$mean_trials), mean(ds_stats$sd_trials)))
    cat(sprintf("  Range: %.0f to %.0f\n",
                min(ds_stats$min_trials), max(ds_stats$max_trials)))
  } else {
    cat("  No data available\n")
  }
}

cat("\n\n=============================================================================\n")
cat("DETAILED STATISTICS BY TIMEBIN (Start-aligned)\n")
cat("=============================================================================\n\n")

for (ds in datasets) {
  cat(sprintf("Dataset: %s\n", ds))
  cat("-----------------------------------------------------------------------------\n")
  ds_stats <- stats_start %>% filter(dataset == ds)
  if (nrow(ds_stats) > 0) {
    print(as.data.frame(ds_stats), row.names = FALSE)
  } else {
    cat("No data available\n")
  }
  cat("\n")
}

cat("\n=============================================================================\n")
cat("DETAILED STATISTICS BY TIMEBIN (Response-aligned)\n")
cat("=============================================================================\n\n")

for (ds in datasets) {
  cat(sprintf("Dataset: %s\n", ds))
  cat("-----------------------------------------------------------------------------\n")
  ds_stats <- stats_response %>% filter(dataset == ds)
  if (nrow(ds_stats) > 0) {
    print(as.data.frame(ds_stats), row.names = FALSE)
  } else {
    cat("No data available\n")
  }
  cat("\n")
}

sink()

message(sprintf("Trial statistics saved to: %s", output_file))
