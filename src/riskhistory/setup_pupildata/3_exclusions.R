# =============================================================================
# Pupil Data Preprocessing Pipeline v2: Apply Exclusion Criteria
# =============================================================================
# Applies thresholds to trial-level statistics and marks trials/subjects
# for exclusion based on data quality.
#
# Key features:
# 1. Dataset-specific thresholds (equalIndifference, equalRange, sheng2020)
# 2. Separate exclusion criteria for start-aligned and response-aligned data
# 3. Adds exclude column to preprocessed pupil data files
# 4. Outputs summary_exclusion.txt with statistics per dataset
# =============================================================================

# Load dependencies
library(dplyr)
library(tidyr)

# Source config
script_dir <- dirname(sys.frame(1)$ofile)
source(file.path(script_dir, "0_config.R"))

# =============================================================================
# Dataset-Specific Exclusion Thresholds
# =============================================================================

# Define thresholds per dataset
thresholds <- list(
  equalIndifference = list(
    pct_nan_baseline = 75,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 2,
    longest_nan_sec_response = 2,
    pct_trials_excluded = 30,
    num_runs = 4
  ),
  equalRange = list(
    pct_nan_baseline = 75,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 2,
    longest_nan_sec_response = 2,
    pct_trials_excluded = 30,
    num_runs = 4
  ),
  sheng2020 = list(
    pct_nan_baseline = 75,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 2,
    longest_nan_sec_response = 2,
    pct_trials_excluded = 30,
    num_runs = 2
  )
)

# =============================================================================
# Setup
# =============================================================================

cat("=== Applying Exclusion Criteria ===\n\n")

cat("Dataset-specific thresholds:\n")
for (ds in names(thresholds)) {
  cat("\n  ", ds, ":\n", sep = "")
  cat("    Max % NaN baseline:", thresholds[[ds]]$pct_nan_baseline, "%\n")
  cat("    Max % NaN start-aligned:", thresholds[[ds]]$pct_nan_start_aligned, "%\n")
  cat("    Max % NaN response-aligned:", thresholds[[ds]]$pct_nan_response_aligned, "%\n")
  cat("    Max longest NaN start-aligned:", thresholds[[ds]]$longest_nan_sec_start, "sec\n")
  cat("    Max longest NaN response-aligned:", thresholds[[ds]]$longest_nan_sec_response, "sec\n")
  cat("    Max % trials excluded:", thresholds[[ds]]$pct_trials_excluded, "%\n")
  cat("    Min unique runs required:", thresholds[[ds]]$num_runs, "\n")
}

# =============================================================================
# Load Statistics Data
# =============================================================================

cat("\n\nLoading statistics data...\n")
statistics <- read.csv(output_statistics)
cat("  Loaded", nrow(statistics), "trial records\n\n")

# =============================================================================
# Apply Trial-Level Exclusions (Dataset-Specific)
# =============================================================================

cat("Applying trial-level exclusions...\n")

# Function to apply thresholds for a specific dataset
# Note: NA values in any statistic are treated as exclusions (TRUE)
apply_trial_exclusions <- function(data, ds_thresholds) {
  data %>%
    mutate(
      # Baseline exclusion (applies to both alignments)
      exclude_baseline = is.na(pct_nan_baseline) | pct_nan_baseline > ds_thresholds$pct_nan_baseline,

      # Start-aligned specific exclusions
      exclude_start_pct = is.na(pct_nan_trial_startAligned) | pct_nan_trial_startAligned > ds_thresholds$pct_nan_start_aligned,
      exclude_start_longest = is.na(longest_nan_sec_startAligned) | longest_nan_sec_startAligned > ds_thresholds$longest_nan_sec_start,

      # Response-aligned specific exclusions
      exclude_response_pct = is.na(pct_nan_trial_responseAligned) | pct_nan_trial_responseAligned > ds_thresholds$pct_nan_response_aligned,
      exclude_response_longest = is.na(longest_nan_sec_responseAligned) | longest_nan_sec_responseAligned > ds_thresholds$longest_nan_sec_response,

      # Separate exclusion flags for each alignment type
      exclude_trial_startAligned = exclude_baseline | exclude_start_pct | exclude_start_longest,
      exclude_trial_responseAligned = exclude_baseline | exclude_response_pct | exclude_response_longest
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
  stats_data %>%
    group_by(sub, dataset) %>%
    summarise(
      n_trials_available = n(),
      n_trials_excluded = sum(.data[[exclude_col]], na.rm = TRUE),
      n_unique_runs = n_distinct(run),
      .groups = "drop"
    ) %>%
    mutate(
      total_trials = case_when(
        dataset == "equalIndifference" ~ NARPS_TOTAL_TRIALS,
        dataset == "equalRange" ~ NARPS_TOTAL_TRIALS,
        dataset == "sheng2020" ~ SHENG2020_TOTAL_TRIALS,
        TRUE ~ NA_integer_
      ),
      n_trials_missing = total_trials - n_trials_available,
      pct_trials_excluded = (n_trials_excluded + n_trials_missing) / total_trials * 100
    ) %>%
    rowwise() %>%
    mutate(
      thresh = thresholds[[dataset]]$pct_trials_excluded,
      exclude_pct_trials = pct_trials_excluded > thresh,
      exclude_num_runs = n_unique_runs < thresholds[[dataset]]$num_runs,
      exclude_subject = exclude_pct_trials | exclude_num_runs
    ) %>%
    ungroup() %>%
    select(sub, dataset, n_trials_available, n_trials_excluded, n_trials_missing,
           total_trials, pct_trials_excluded, n_unique_runs,
           exclude_pct_trials, exclude_num_runs, exclude_subject) %>%
    rename_with(~ paste0(., "_", alignment_name),
                .cols = c("n_trials_excluded", "pct_trials_excluded",
                          "exclude_pct_trials", "exclude_num_runs", "exclude_subject"))
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
  n_excl_runs <- sum(ds_data$exclude_num_runs_startAligned)
  n_excl_start <- sum(ds_data$exclude_subject_startAligned)
  n_excl_response <- sum(ds_data$exclude_subject_responseAligned)
  cat("\n  ", ds, ":\n", sep = "")
  cat("    Insufficient runs:", n_excl_runs, "/", n_total, "subjects excluded\n")
  cat("    Start-aligned:", n_excl_start, "/", n_total, "subjects excluded (total)\n")
  cat("    Response-aligned:", n_excl_response, "/", n_total, "subjects excluded (total)\n")
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
cat("  Updated:", output_start_aligned, "\n")
cat("  Rows excluded:", sum(start_aligned$exclude), "/", nrow(start_aligned), "\n")

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
cat("  Updated:", output_response_aligned, "\n")
cat("  Rows excluded:", sum(response_aligned$exclude), "/", nrow(response_aligned), "\n")

# =============================================================================
# Generate Summary Exclusion Report
# =============================================================================

cat("\n=== Generating summary exclusion report ===\n")

output_summary <- file.path(results_path, "setup_pupildata", "summary_exclusion.txt")

# Calculate summary statistics
summary_lines <- c(
  "=============================================================================",
  "Pupil Data Exclusion Summary",
  paste("Generated:", Sys.time()),
  "=============================================================================",
  ""
)

for (ds in unique(statistics$dataset)) {
  ds_stats <- statistics %>% filter(dataset == ds)
  ds_subj <- subject_exclusions %>% filter(dataset == ds)

  # Start-aligned statistics
  n_subj_total_start <- nrow(ds_subj)
  n_subj_excluded_start <- sum(ds_subj$exclude_subject_startAligned)
  n_subj_included_start <- n_subj_total_start - n_subj_excluded_start

  # Average trials per included subject (start-aligned)
  included_subs_start <- ds_subj %>%
    filter(!exclude_subject_startAligned) %>%
    pull(sub)

  avg_trials_start <- ds_stats %>%
    filter(sub %in% included_subs_start, !exclude_trial_startAligned) %>%
    group_by(sub) %>%
    summarise(n = n(), .groups = "drop") %>%
    summarise(mean_trials = mean(n)) %>%
    pull(mean_trials)

  # Response-aligned statistics
  n_subj_excluded_response <- sum(ds_subj$exclude_subject_responseAligned)
  n_subj_included_response <- n_subj_total_start - n_subj_excluded_response

  included_subs_response <- ds_subj %>%
    filter(!exclude_subject_responseAligned) %>%
    pull(sub)

  avg_trials_response <- ds_stats %>%
    filter(sub %in% included_subs_response, !exclude_trial_responseAligned) %>%
    group_by(sub) %>%
    summarise(n = n(), .groups = "drop") %>%
    summarise(mean_trials = mean(n)) %>%
    pull(mean_trials)

  summary_lines <- c(summary_lines,
    paste0("Dataset: ", ds),
    "-----------------------------------------------------------------------------",
    "",
    "Start-Aligned:",
    paste0("  Subjects excluded: ", n_subj_excluded_start, " / ", n_subj_total_start),
    paste0("  Subjects included: ", n_subj_included_start),
    paste0("  Average trials per included subject: ", round(avg_trials_start, 1)),
    "",
    "Response-Aligned:",
    paste0("  Subjects excluded: ", n_subj_excluded_response, " / ", n_subj_total_start),
    paste0("  Subjects included: ", n_subj_included_response),
    paste0("  Average trials per included subject: ", round(avg_trials_response, 1)),
    "",
    ""
  )
}

# Add thresholds used
summary_lines <- c(summary_lines,
  "=============================================================================",
  "Thresholds Used",
  "=============================================================================",
  ""
)

for (ds in names(thresholds)) {
  summary_lines <- c(summary_lines,
    paste0("Dataset: ", ds),
    paste0("  Max % NaN baseline: ", thresholds[[ds]]$pct_nan_baseline, "%"),
    paste0("  Max % NaN start-aligned: ", thresholds[[ds]]$pct_nan_start_aligned, "%"),
    paste0("  Max % NaN response-aligned: ", thresholds[[ds]]$pct_nan_response_aligned, "%"),
    paste0("  Max longest NaN start-aligned: ", thresholds[[ds]]$longest_nan_sec_start, " sec"),
    paste0("  Max longest NaN response-aligned: ", thresholds[[ds]]$longest_nan_sec_response, " sec"),
    paste0("  Max % trials excluded per subject: ", thresholds[[ds]]$pct_trials_excluded, "%"),
    paste0("  Min unique runs required: ", thresholds[[ds]]$num_runs),
    ""
  )
}

writeLines(summary_lines, output_summary)
cat("Saved:", output_summary, "\n")

# =============================================================================
# Final Console Summary
# =============================================================================

cat("\n=== Final Summary ===\n")

for (ds in unique(statistics$dataset)) {
  ds_subj <- subject_exclusions %>% filter(dataset == ds)
  ds_stats <- statistics %>% filter(dataset == ds)

  n_subj <- nrow(ds_subj)
  n_excl_start <- sum(ds_subj$exclude_subject_startAligned)
  n_excl_response <- sum(ds_subj$exclude_subject_responseAligned)

  cat("\n", ds, ":\n", sep = "")
  cat("  Start-aligned: ", n_subj - n_excl_start, "/", n_subj, " subjects included\n", sep = "")
  cat("  Response-aligned: ", n_subj - n_excl_response, "/", n_subj, " subjects included\n", sep = "")
}

cat("\n=== Exclusion criteria applied ===\n")
