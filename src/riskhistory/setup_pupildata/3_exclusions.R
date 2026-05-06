# =============================================================================
# Pupil Data Preprocessing Pipeline: Apply Exclusion Criteria
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
# Dataset-Specific Exclusion Thresholds
# =============================================================================

# Define thresholds per dataset
thresholds <- list(
  equalIndifference = list(
    pct_nan_baseline = 100,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 2,
    longest_nan_sec_response = 2,
    pct_trials_excluded = 30,
    num_runs = 4
  ),
  equalRange = list(
    pct_nan_baseline = 100,
    pct_nan_start_aligned = 40,
    pct_nan_response_aligned = 40,
    longest_nan_sec_start = 2,
    longest_nan_sec_response = 2,
    pct_trials_excluded = 30,
    num_runs = 4
  ),
  sheng2020 = list(
    pct_nan_baseline = 100,
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

cat("Loading behavioral exclusion flags...\n")
behavioral_exclusions <- read.csv(behavioral_data_path) %>%
  select(sub, trial, dataset, exclude_behavioral = exclude)
behavioral_exclusions <- behavioral_exclusions %>%
  mutate(exclude_behavioral = as.logical(exclude_behavioral))
cat("  Loaded", nrow(behavioral_exclusions), "behavioral records\n\n")

# Merge behavioral exclusions into statistics
statistics <- statistics %>%
  left_join(behavioral_exclusions, by = c("sub", "trial", "dataset"))

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

      # Behavioral exclusion (from behavioral preprocessing)
      exclude_behavioral = exclude_behavioral,

      # Separate exclusion flags for each alignment type
      exclude_trial_startAligned = exclude_baseline | exclude_start_pct | exclude_start_longest | exclude_behavioral,
      exclude_trial_responseAligned = exclude_baseline | exclude_response_pct | exclude_response_longest | exclude_behavioral
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
# Compute Detailed Trial-Level Exclusion Counts
# =============================================================================

# Base-included subjects: all expected runs present AND not subject-level
# behaviorally excluded (subjects in behavioral_data.csv passed subject-level
# behavioral criteria; trial-level exclude_behavioral flags remain).
behavioral_subjects <- behavioral_exclusions %>%
  select(sub, dataset) %>%
  distinct()

subjects_all_runs <- subject_exclusions %>%
  filter(!exclude_num_runs_startAligned) %>%
  select(sub, dataset)

base_included_subjects <- subjects_all_runs %>%
  semi_join(behavioral_subjects, by = c("sub", "dataset"))

cat("\nBase-included subjects (all runs + behavioral):", nrow(base_included_subjects), "\n")

stats_base_included <- statistics %>%
  semi_join(base_included_subjects, by = c("sub", "dataset")) %>%
  mutate(exclude_behavioral = coalesce(as.logical(exclude_behavioral), FALSE))

# Marginal trial counts — start-aligned
# Applied order: behavioral -> baseline -> % NaN in window -> longest NaN gap
marginal_start <- stats_base_included %>%
  group_by(sub, dataset) %>%
  summarise(
    n_behavioral    = sum(exclude_behavioral, na.rm = TRUE),
    n_baseline      = sum(!exclude_behavioral & exclude_baseline, na.rm = TRUE),
    n_start_pct     = sum(!exclude_behavioral & !exclude_baseline & exclude_start_pct, na.rm = TRUE),
    n_start_longest = sum(!exclude_behavioral & !exclude_baseline & !exclude_start_pct & exclude_start_longest, na.rm = TRUE),
    n_total         = sum(exclude_trial_startAligned, na.rm = TRUE),
    .groups = "drop"
  )

# Marginal trial counts — response-aligned
marginal_response <- stats_base_included %>%
  group_by(sub, dataset) %>%
  summarise(
    n_behavioral       = sum(exclude_behavioral, na.rm = TRUE),
    n_baseline         = sum(!exclude_behavioral & exclude_baseline, na.rm = TRUE),
    n_response_pct     = sum(!exclude_behavioral & !exclude_baseline & exclude_response_pct, na.rm = TRUE),
    n_response_longest = sum(!exclude_behavioral & !exclude_baseline & !exclude_response_pct & exclude_response_longest, na.rm = TRUE),
    n_total            = sum(exclude_trial_responseAligned, na.rm = TRUE),
    .groups = "drop"
  )

format_criterion_row <- function(label, values) {
  sprintf("    %-32s M = %5.1f  SD = %5.1f  Total = %d",
          label,
          mean(values, na.rm = TRUE),
          sd(values, na.rm = TRUE),
          as.integer(sum(values, na.rm = TRUE)))
}

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

# -------------------------------------------------------------------------
# Trial-level marginal exclusion counts
# -------------------------------------------------------------------------
summary_lines <- c(summary_lines,
  "",
  "=============================================================================",
  "Trial-Level Marginal Exclusion Counts (base-included subjects only)",
  "  Base-included: all expected runs present AND not subject-level behaviorally excluded",
  "  Applied order: behavioral -> baseline -> % NaN in window -> longest NaN gap",
  "=============================================================================",
  ""
)

for (ds in unique(statistics$dataset)) {
  ds_start    <- marginal_start    %>% filter(dataset == ds)
  ds_response <- marginal_response %>% filter(dataset == ds)

  summary_lines <- c(summary_lines,
    paste0("Dataset: ", ds),
    "",
    paste0("  Start-Aligned (N = ", nrow(ds_start), " subjects):"),
    format_criterion_row("behavioral exclusion",  ds_start$n_behavioral),
    format_criterion_row("baseline % NaN",        ds_start$n_baseline),
    format_criterion_row("task window % NaN",     ds_start$n_start_pct),
    format_criterion_row("longest NaN gap",       ds_start$n_start_longest),
    format_criterion_row("TOTAL",                 ds_start$n_total),
    "",
    paste0("  Response-Aligned (N = ", nrow(ds_response), " subjects):"),
    format_criterion_row("behavioral exclusion",  ds_response$n_behavioral),
    format_criterion_row("baseline % NaN",        ds_response$n_baseline),
    format_criterion_row("task window % NaN",     ds_response$n_response_pct),
    format_criterion_row("longest NaN gap",       ds_response$n_response_longest),
    format_criterion_row("TOTAL",                 ds_response$n_total),
    ""
  )
}

# -------------------------------------------------------------------------
# Subject-level sole-criterion exclusion counts
# -------------------------------------------------------------------------
summary_lines <- c(summary_lines,
  "",
  "=============================================================================",
  "Subject-Level Exclusion Counts (sole criterion)",
  "=============================================================================",
  ""
)

for (ds in unique(statistics$dataset)) {
  ds_subj      <- subject_exclusions %>% filter(dataset == ds)
  n_total_subs <- nrow(ds_subj)

  only_runs_start <- sum(!ds_subj$exclude_pct_trials_startAligned &  ds_subj$exclude_num_runs_startAligned,  na.rm = TRUE)
  only_pct_start  <- sum( ds_subj$exclude_pct_trials_startAligned & !ds_subj$exclude_num_runs_startAligned,  na.rm = TRUE)
  both_start      <- sum( ds_subj$exclude_pct_trials_startAligned &  ds_subj$exclude_num_runs_startAligned,  na.rm = TRUE)
  total_start     <- sum( ds_subj$exclude_subject_startAligned,  na.rm = TRUE)

  only_runs_resp  <- sum(!ds_subj$exclude_pct_trials_responseAligned &  ds_subj$exclude_num_runs_responseAligned, na.rm = TRUE)
  only_pct_resp   <- sum( ds_subj$exclude_pct_trials_responseAligned & !ds_subj$exclude_num_runs_responseAligned, na.rm = TRUE)
  both_resp       <- sum( ds_subj$exclude_pct_trials_responseAligned &  ds_subj$exclude_num_runs_responseAligned, na.rm = TRUE)
  total_resp      <- sum( ds_subj$exclude_subject_responseAligned, na.rm = TRUE)

  summary_lines <- c(summary_lines,
    paste0("Dataset: ", ds, " (N = ", n_total_subs, " subjects)"),
    "",
    "  Start-Aligned:",
    paste0("    Insufficient runs (sole):       ", only_runs_start),
    paste0("    Too many excl. trials (sole):   ", only_pct_start),
    paste0("    Both criteria:                  ", both_start),
    paste0("    Total excluded:                 ", total_start),
    "",
    "  Response-Aligned:",
    paste0("    Insufficient runs (sole):       ", only_runs_resp),
    paste0("    Too many excl. trials (sole):   ", only_pct_resp),
    paste0("    Both criteria:                  ", both_resp),
    paste0("    Total excluded:                 ", total_resp),
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
