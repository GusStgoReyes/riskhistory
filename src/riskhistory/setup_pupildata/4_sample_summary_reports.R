# =============================================================================
# Pupil Data Preprocessing Pipeline: Sample & Exclusion Summary Reports
# =============================================================================
# Standalone script -- NOT part of run_setup_pupildata.R's step chain. Run
# manually (Rscript 4_sample_summary_reports.R from this directory) after the
# main pipeline has already produced, via 3_exclusions.R:
#   results/setup_pupildata/pupil_data_statistics_with_exclusions.csv
#   results/setup_pupildata/pupil_subject_exclusions.csv
#
# Mirrors the 3-file reporting convention from setup_behavioraldata
# (final_sample_summary.csv, participant_exclusion_summary.csv,
# trial_exclusion_summary.csv), but for pupil-specific exclusion criteria
# (NaN%, longest-gap, insufficient-runs) layered on top of the already
# behaviorally-clean sample. Start-aligned only.
# =============================================================================

library(dplyr)
source("0_config.R")

trial_stats_path <- file.path(results_path, "setup_pupildata",
                              "pupil_data_statistics_with_exclusions.csv")
subject_exclusions_path <- file.path(results_path, "setup_pupildata",
                                     "pupil_subject_exclusions.csv")

if (!file.exists(trial_stats_path) || !file.exists(subject_exclusions_path)) {
  stop(
    "Missing prerequisite file(s). Run the main pupil pipeline through ",
    "3_exclusions.R first (e.g. `Rscript run_setup_pupildata.R`)."
  )
}

trial_stats <- read.csv(trial_stats_path) %>%
  mutate(across(starts_with("exclude"), as.logical))
subject_exclusions <- read.csv(subject_exclusions_path) %>%
  mutate(across(starts_with("exclude"), as.logical))

behavioral_data <- read.csv(behavioral_data_path) %>%
  select(sub, trial, dataset, accept, rt)

# =============================================================================
# 1. Participant exclusion summary
# =============================================================================
# "Started off with" = subjects with pupil data who already passed behavioral
# inclusion (the pool 3_exclusions.R starts pupil-specific exclusion from).
# Criteria counted independently (a subject failing both is counted under
# each), matching setup_behavioraldata's convention.
message("Building participant exclusion summary...")
participant_exclusion_summary <- subject_exclusions %>%
  group_by(dataset) %>%
  summarise(
    n_subjects = n(),
    n_excluded_insufficient_runs = sum(exclude_num_runs_startAligned),
    n_excluded_high_trial_exclusion_pct = sum(exclude_pct_trials_startAligned),
    n_excluded_total = sum(exclude_subject_startAligned),
    .groups = "drop"
  )

# =============================================================================
# 2. Trial exclusion summary
# =============================================================================
# Per-subject trial-level exclusion counts (independent criteria), before
# subject-level exclusion, then averaged to dataset level.
message("Building trial exclusion summary...")
trial_exclusion_counts <- trial_stats %>%
  group_by(dataset, sub) %>%
  summarise(
    n_trials = n(),
    n_excluded_behavioral = sum(exclude_behavioral, na.rm = TRUE),
    n_excluded_baseline = sum(exclude_baseline, na.rm = TRUE),
    n_excluded_pct_nan = sum(exclude_start_pct, na.rm = TRUE),
    n_excluded_longest_nan = sum(exclude_start_longest, na.rm = TRUE),
    n_excluded_total = sum(exclude_trial_startAligned, na.rm = TRUE),
    .groups = "drop"
  )

trial_exclusion_summary <- trial_exclusion_counts %>%
  group_by(dataset) %>%
  summarise(
    n_subjects = n(),
    mean_n_excluded_behavioral = mean(n_excluded_behavioral),
    sd_n_excluded_behavioral = sd(n_excluded_behavioral),
    mean_n_excluded_baseline = mean(n_excluded_baseline),
    sd_n_excluded_baseline = sd(n_excluded_baseline),
    mean_n_excluded_pct_nan = mean(n_excluded_pct_nan),
    sd_n_excluded_pct_nan = sd(n_excluded_pct_nan),
    mean_n_excluded_longest_nan = mean(n_excluded_longest_nan),
    sd_n_excluded_longest_nan = sd(n_excluded_longest_nan),
    mean_n_excluded_total = mean(n_excluded_total),
    sd_n_excluded_total = sd(n_excluded_total),
    .groups = "drop"
  )

# =============================================================================
# 3. Final sample summary
# =============================================================================
# Subjects retained after pupil-specific subject-level exclusion, and trials
# passing the full (behavioral + pupil) start-aligned exclusion, with
# accept/rt merged in from behavioral data.
message("Building final sample summary...")
retained_subjects <- subject_exclusions %>%
  filter(!exclude_subject_startAligned) %>%
  select(sub, dataset)

valid_trials <- trial_stats %>%
  filter(!exclude_startAligned) %>%
  semi_join(retained_subjects, by = c("sub", "dataset")) %>%
  select(sub, trial, dataset) %>%
  left_join(behavioral_data, by = c("sub", "trial", "dataset"))

subject_final_stats <- valid_trials %>%
  group_by(dataset, sub) %>%
  summarise(
    n_valid_trials = n(),
    subject_acceptance_rate = mean(accept, na.rm = TRUE),
    subject_mean_rt = mean(rt, na.rm = TRUE),
    .groups = "drop"
  )

final_sample_summary <- subject_final_stats %>%
  group_by(dataset) %>%
  summarise(
    n_subjects = n(),
    mean_n_valid_trials = mean(n_valid_trials),
    min_n_valid_trials = min(n_valid_trials),
    max_n_valid_trials = max(n_valid_trials),
    mean_acceptance_rate = mean(subject_acceptance_rate, na.rm = TRUE),
    sd_acceptance_rate = sd(subject_acceptance_rate, na.rm = TRUE),
    mean_rt = mean(subject_mean_rt, na.rm = TRUE),
    sd_rt = sd(subject_mean_rt, na.rm = TRUE),
    .groups = "drop"
  )

# =============================================================================
# Save outputs
# =============================================================================
output_dir <- file.path(results_path, "setup_pupildata")

write.csv(participant_exclusion_summary,
          file.path(output_dir, "participant_exclusion_summary.csv"),
          row.names = FALSE)
message("Saved: participant_exclusion_summary.csv")

write.csv(trial_exclusion_summary,
          file.path(output_dir, "trial_exclusion_summary.csv"),
          row.names = FALSE)
message("Saved: trial_exclusion_summary.csv")

write.csv(final_sample_summary,
          file.path(output_dir, "final_sample_summary.csv"),
          row.names = FALSE)
message("Saved: final_sample_summary.csv")

message("\nPupil sample/exclusion summary reports complete.")
