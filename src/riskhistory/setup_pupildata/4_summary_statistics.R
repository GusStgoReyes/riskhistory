# =============================================================================
# Pupil Data Preprocessing Pipeline: Summary Statistics
# =============================================================================
# Computes summary statistics for participants included in the start-aligned
# pupil analysis: mean acceptance rate, number of valid trials, and mean RT.
#
# Trial-level exclusions (including behavioral) are captured in exclude_startAligned
# from 3_exclusions.R.
#
# Outputs:
#   results/setup_pupildata/behavioral_summary.txt  (printed summary text)
# =============================================================================

library(dplyr)

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

cat("=== Computing Summary Statistics for Included Participants ===\n\n")

# =============================================================================
# Load Data
# =============================================================================

cat("Loading behavioral data...\n")
behavioral_data <- read.csv(behavioral_data_path)
cat("  Loaded", nrow(behavioral_data), "rows\n")

cat("Loading trial-level start-aligned exclusions...\n")
trial_exclusions_start <- read.csv(
  file.path(results_path, "setup_pupildata", "pupil_data_statistics_with_exclusions.csv")
) %>%
  select(sub, trial, dataset, exclude_startAligned) %>%
  distinct()
cat("  Loaded", nrow(trial_exclusions_start), "trials\n\n")

# =============================================================================
# Filter to Included Participants (Both Criteria)
# =============================================================================

behavioral_subjects <- behavioral_data %>%
  select(sub, dataset) %>%
  distinct()

cat("Subjects in behavioral_data:\n")
for (ds in unique(behavioral_subjects$dataset)) {
  n <- sum(behavioral_subjects$dataset == ds)
  cat("  ", ds, ":", n, "subjects\n")
}
cat("\n")


# =============================================================================
# Compute Per-Subject Statistics
# =============================================================================

# Keep only included subjects and non-excluded start-aligned trials
# (exclude_startAligned already incorporates behavioral exclusions)
data_included <- behavioral_data %>%
  left_join(trial_exclusions_start, by = c("sub", "trial", "dataset")) %>%
  filter(!exclude_startAligned)

per_subject_stats <- data_included %>%
  group_by(sub, dataset) %>%
  summarise(
    n_trials      = n(),
    acceptance_rate = mean(accept, na.rm = TRUE),
    avg_rt       = mean(rt, na.rm = TRUE),
    .groups = "drop"
  )

# =============================================================================
# Compute Dataset-Level Summary
# =============================================================================

dataset_summary <- per_subject_stats %>%
  group_by(dataset) %>%
  summarise(
    n_subjects           = n(),
    mean_n_trials        = mean(n_trials),
    sd_n_trials          = sd(n_trials),
    min_n_trials         = min(n_trials),
    max_n_trials         = max(n_trials),
    mean_acceptance_rate = mean(acceptance_rate),
    sd_acceptance_rate   = sd(acceptance_rate),
    mean_rt              = mean(avg_rt),
    sd_rt                = sd(avg_rt),
    .groups = "drop"
  )

summary_lines <- c("Summary statistics by dataset:")
for (i in seq_len(nrow(dataset_summary))) {
  ds <- dataset_summary$dataset[i]
  summary_lines <- c(summary_lines,
    paste0("\n  ", ds, ":"),
    paste0("    N subjects:       ", dataset_summary$n_subjects[i]),
    paste0("    Mean trials:      ", round(dataset_summary$mean_n_trials[i], 1),
           " (SD = ", round(dataset_summary$sd_n_trials[i], 1),
           ", range = [", dataset_summary$min_n_trials[i], ", ",
           dataset_summary$max_n_trials[i], "])"),
    paste0("    Acceptance rate:  ", round(dataset_summary$mean_acceptance_rate[i], 3),
           " (SD = ", round(dataset_summary$sd_acceptance_rate[i], 3), ")"),
    paste0("    Mean RT (s):      ", round(dataset_summary$mean_rt[i], 3),
           " (SD = ", round(dataset_summary$sd_rt[i], 3), ")")
  )
}

cat(paste(summary_lines, collapse = "\n"), "\n")

# =============================================================================
# Save Outputs
# =============================================================================

output_dir <- file.path(results_path, "setup_pupildata")
ensure_dir(output_dir)

output_txt <- file.path(output_dir, "behavioral_summary.txt")
writeLines(summary_lines, output_txt)
cat("\nSaved:", output_txt, "\n")

cat("\n=== Summary statistics complete ===\n")
