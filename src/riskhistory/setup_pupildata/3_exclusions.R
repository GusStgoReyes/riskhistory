# =============================================================================
# Pupil Data Preprocessing Pipeline: Apply Exclusion Criteria
# =============================================================================

# Load dependencies
library(dplyr)
library(tidyr)
source("0_config.R")

# =============================================================================
# Dataset-Specific Exclusion Thresholds
# =============================================================================

# Define thresholds per dataset
thresholds <- list(
  equalIndifference = list(
    pct_nan_baseline = 100,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 4,
    longest_nan_sec_response = 4,
    pct_trials_excluded = 30,
    num_runs = 4
  ),
  equalRange = list(
    pct_nan_baseline = 100,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 4,
    longest_nan_sec_response = 4,
    pct_trials_excluded = 30,
    num_runs = 4
  ),
  sheng2020 = list(
    pct_nan_baseline = 100,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 4,
    longest_nan_sec_response = 4,
    pct_trials_excluded = 30,
    num_runs = 2
  )
)

cat("=== Applying Exclusion Criteria ===\n\n")

# =============================================================================
# Load Statistics Data
# =============================================================================

cat("\n\nLoading statistics data...\n")
statistics <- read.csv(output_statistics)
cat("  Loaded", nrow(statistics), "trial records\n\n")

cat("Loading behavioral exclusion flags...\n")
behavioral_exclusions <- read.csv(behavioral_data_path) %>%
  select(sub, trial, dataset, exclude_behavioral = exclude)
behavioral_exclusions <- behavioral_exclusions %>%
  mutate(exclude_behavioral = as.logical(exclude_behavioral))
behavioral_subjects <- behavioral_exclusions %>%
  filter(!exclude_behavioral) %>%
  select(sub, dataset) %>%
  distinct()
  
cat(" Mean trials per subject after behavioral exclusions (by dataset):\n\n")
behavioral_exclusions %>%
  filter(!exclude_behavioral) %>%
  group_by(dataset, sub) %>%
  summarise(n_trials = n(), .groups = "drop") %>%
  group_by(dataset) %>%
  summarise(mean_trials = mean(n_trials), sd_trials = sd(n_trials), n_subs = n_distinct(sub), .groups = "drop") %>%
  print()

statistics <- statistics %>%
  semi_join(behavioral_subjects, by = c("sub", "dataset"))
pupil_subjects <- statistics %>%
  select(sub, dataset) %>%
  distinct()
cat("\n Subjects with pupil data after behavioral exclusions (by dataset):\n\n")
pupil_subjects %>%
  group_by(dataset) %>%
  summarise(n_subs = n(), .groups = "drop") %>%
  print()

# Merge trial-level behavioral exclusion flags
statistics <- statistics %>%
  left_join(behavioral_exclusions, by = c("sub", "trial", "dataset"))

# =============================================================================
# Apply Trial-Level Exclusions (Dataset-Specific)
# =============================================================================

cat("Applying trial-level exclusions...\n")
apply_trial_exclusions <- function(data, ds_thresholds) {
  data %>%
    mutate(
      # Baseline exclusion (applies to both alignments)
      exclude_baseline = (is.na(pct_nan_baseline) |
                            pct_nan_baseline > ds_thresholds$pct_nan_baseline),

      # Start-aligned specific exclusions
      exclude_start_pct = (is.na(pct_nan_trial_startAligned) |
                             pct_nan_trial_startAligned > ds_thresholds$pct_nan_start_aligned),
      exclude_start_longest = (is.na(longest_nan_sec_startAligned) |
                                 longest_nan_sec_startAligned > ds_thresholds$longest_nan_sec_start),

      # Response-aligned specific exclusions
      exclude_response_pct = (is.na(pct_nan_trial_responseAligned) |
                                pct_nan_trial_responseAligned > ds_thresholds$pct_nan_response_aligned),
      exclude_response_longest = (is.na(longest_nan_sec_responseAligned) |
                                    longest_nan_sec_responseAligned > ds_thresholds$longest_nan_sec_response) ,

      # Behavioral exclusion (from behavioral preprocessing)
      exclude_behavioral = exclude_behavioral,

      # Separate exclusion flags for each alignment type
      exclude_trial_startAligned = (exclude_baseline |
                                      exclude_start_pct |
                                      exclude_start_longest |
                                      exclude_behavioral),
      exclude_trial_responseAligned = (exclude_baseline |
                                         exclude_response_pct |
                                         exclude_response_longest |
                                         exclude_behavioral)
    )
}

# Apply exclusions per dataset
statistics <- statistics %>%
  group_by(dataset) %>%
  group_modify(~ {
    ds <- .y$dataset[1]
    if (ds %in% names(thresholds)) {
      apply_trial_exclusions(.x, thresholds[[ds]])
    } else {
      warning(paste("Unknown dataset:", ds, "- using equalIndifference thresholds"))
      apply_trial_exclusions(.x, thresholds$equalIndifference)
    }
  }) %>%
  ungroup()

# =============================================================================
# Apply Subject-Level Exclusions (Separate for Each Alignment)
# =============================================================================

cat("\n\nApplying subject-level exclusions...\n")

# Function to calculate subject exclusions for a specific alignment
calculate_subject_exclusions <- function(stats_data, exclude_col, alignment_name) {
  threshold_lut <- tibble(
    dataset          = names(thresholds),
    thresh_pct_trial = sapply(thresholds, `[[`, "pct_trials_excluded"),
    thresh_num_runs  = sapply(thresholds, `[[`, "num_runs")
  )

  stats_data %>%
    group_by(sub, dataset) %>%
    summarise(
      n_trials_available = n(),
      n_trials_excluded  = sum(.data[[exclude_col]], na.rm = TRUE),
      n_unique_runs      = n_distinct(run),
      .groups = "drop"
    ) %>%
    mutate(
      total_trials = case_when(
        dataset == "equalIndifference" ~ NARPS_TOTAL_TRIALS,
        dataset == "equalRange"        ~ NARPS_TOTAL_TRIALS,
        dataset == "sheng2020"         ~ SHENG2020_TOTAL_TRIALS,
        TRUE ~ NA_integer_
      ),
      n_trials_missing    = total_trials - n_trials_available,
      pct_trials_excluded = (n_trials_excluded + n_trials_missing) / total_trials * 100
    ) %>%
    left_join(threshold_lut, by = "dataset") %>%
    mutate(
      exclude_pct_trials = pct_trials_excluded > thresh_pct_trial,
      exclude_num_runs   = n_unique_runs < thresh_num_runs,
      exclude_subject    = exclude_pct_trials | exclude_num_runs
    ) %>%
    select(-thresh_pct_trial, -thresh_num_runs) %>%
    select(sub, dataset, n_trials_available, n_trials_excluded, n_trials_missing,
           total_trials, pct_trials_excluded, n_unique_runs,
           exclude_pct_trials, exclude_num_runs, exclude_subject) %>%
    rename_with(
      ~ paste0(., "_", alignment_name),
      .cols = c("n_trials_excluded", "pct_trials_excluded",
                "exclude_pct_trials", "exclude_num_runs", "exclude_subject")
    )
}

# Calculate subject exclusions for each alignment
subject_exclusions_start <- calculate_subject_exclusions(
  statistics, "exclude_trial_startAligned", "startAligned"
)
subject_exclusions_response <- calculate_subject_exclusions(
  statistics, "exclude_trial_responseAligned", "responseAligned"
)

# Merge the two
subject_exclusions <- subject_exclusions_start %>%
  left_join(
    subject_exclusions_response %>%
      select(sub, dataset, ends_with("_responseAligned")),
    by = c("sub", "dataset")
  )

# Summary by dataset
cat("\nSubject exclusions by dataset and alignment:\n")
for (ds in unique(subject_exclusions$dataset)) {
  ds_data <- subject_exclusions %>% filter(dataset == ds)
  n_total <- nrow(ds_data)

  runs_s <- ds_data$exclude_num_runs_startAligned
  pct_s  <- ds_data$exclude_pct_trials_startAligned
  n_runs_start      <- sum(runs_s, na.rm = TRUE)
  n_pct_only_start  <- sum(!runs_s & pct_s, na.rm = TRUE)
  n_total_start     <- sum(ds_data$exclude_subject_startAligned, na.rm = TRUE)

  runs_r <- ds_data$exclude_num_runs_responseAligned
  pct_r  <- ds_data$exclude_pct_trials_responseAligned
  n_runs_resp       <- sum(runs_r, na.rm = TRUE)
  n_pct_only_resp   <- sum(!runs_r & pct_r, na.rm = TRUE)
  n_total_resp      <- sum(ds_data$exclude_subject_responseAligned, na.rm = TRUE)

  cat("\n  ", ds, " (N = ", n_total, "):\n", sep = "")
  cat("    Start-aligned:\n")
  cat("      Insufficient runs:                    ", n_runs_start, "\n")
  cat("      Too many excl. trials (runs ok):      ", n_pct_only_start, "\n")
  cat("      Total excluded:                       ", n_total_start, "/", n_total, "\n")
  cat("    Response-aligned:\n")
  cat("      Insufficient runs:                    ", n_runs_resp, "\n")
  cat("      Too many excl. trials (runs ok):      ", n_pct_only_resp, "\n")
  cat("      Total excluded:                       ", n_total_resp, "/", n_total, "\n")
}

# =============================================================================
# Merge Subject Exclusion Back to Trial Data
# =============================================================================

statistics <- statistics %>%
  left_join(
    subject_exclusions %>%
      select(sub, dataset, exclude_subject_startAligned, exclude_subject_responseAligned),
    by = c("sub", "dataset")
  ) %>%
  mutate(
    # Final exclusion flags for each alignment
    exclude_startAligned = exclude_trial_startAligned | exclude_subject_startAligned,
    exclude_responseAligned = exclude_trial_responseAligned | exclude_subject_responseAligned
  )

cat("\nMean trials removed per subject by criterion (start-aligned,",
    "included subjects only):\n")
statistics %>%
  filter(!exclude_subject_startAligned) %>%
  group_by(dataset, sub) %>%
  summarise(
    n_behavioral    = sum(exclude_behavioral,    na.rm = TRUE),
    n_start_pct     = sum(exclude_start_pct,     na.rm = TRUE),
    n_baseline      = sum(exclude_baseline,       na.rm = TRUE),
    n_start_longest = sum(exclude_start_longest,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(dataset) %>%
  summarise(
    mean_behavioral    = mean(n_behavioral),
    mean_start_pct     = mean(n_start_pct),
    mean_baseline      = mean(n_baseline),
    mean_start_longest = mean(n_start_longest),
    .groups = "drop"
  ) %>%
  print()

# =============================================================================
# Save Statistics Results
# =============================================================================

cat("\n\n=== Saving exclusion results ===\n")

# Save updated statistics with exclusion columns
output_statistics_with_exclusions <- file.path(
  results_path, "setup_pupildata", "pupil_data_statistics_with_exclusions.csv"
)
ensure_dir(dirname(output_statistics_with_exclusions))

write.csv(
  statistics,
  file = output_statistics_with_exclusions,
  row.names = FALSE
)
cat("Saved:", output_statistics_with_exclusions, "\n")

# Save subject-level exclusion summary
output_subject_exclusions <- file.path(
  results_path, "setup_pupildata", "pupil_subject_exclusions.csv"
)

write.csv(
  subject_exclusions,
  file = output_subject_exclusions,
  row.names = FALSE
)
cat("Saved:", output_subject_exclusions, "\n")

# =============================================================================
# Add Exclude Column to Preprocessed Pupil Data
# =============================================================================

cat("\n=== Adding exclude column to preprocessed pupil data ===\n")

# Load start-aligned pupil data
cat("Processing start-aligned data...\n")
start_aligned <- read.csv(output_start_aligned)

# Create exclusion lookup for start-aligned
exclusion_lookup_start <- statistics %>%
  select(sub, trial, dataset, exclude_startAligned) %>%
  distinct()

# Merge exclusion flags
start_aligned <- start_aligned %>%
  left_join(
    exclusion_lookup_start,
    by = c("sub", "trial", "dataset")
  ) %>%
  mutate(
    # Exclude if subject is excluded OR trial is excluded
    exclude = exclude_startAligned
  ) %>%
  select(-exclude_startAligned)

write.csv(start_aligned, file = output_start_aligned, row.names = FALSE)

# Load response-aligned pupil data
cat("\nProcessing response-aligned data...\n")
response_aligned <- read.csv(output_response_aligned)

# Create exclusion lookup for response-aligned
exclusion_lookup_response <- statistics %>%
  select(sub, trial, dataset, exclude_responseAligned) %>%
  distinct()

# Merge exclusion flags
response_aligned <- response_aligned %>%
  left_join(
    exclusion_lookup_response,
    by = c("sub", "trial", "dataset")
  ) %>%
  mutate(
    exclude = exclude_responseAligned
  ) %>%
  select(-exclude_responseAligned)

write.csv(response_aligned, file = output_response_aligned, row.names = FALSE)