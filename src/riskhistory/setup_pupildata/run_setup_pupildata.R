#!/usr/bin/env Rscript
# =============================================================================
# Pupil Data Preprocessing Pipeline v2: Main Runner
# =============================================================================
# Runs the complete pupil preprocessing pipeline for both datasets.
#
# Usage:
#   Rscript run_setup_pupildata.R              # Run all steps
#   Rscript run_setup_pupildata.R narps        # Run only NARPS preprocessing
#   Rscript run_setup_pupildata.R sheng2020    # Run only Sheng2020 preprocessing
#   Rscript run_setup_pupildata.R combine      # Run only the combine step
#   Rscript run_setup_pupildata.R exclusions   # Run only the exclusions step
# =============================================================================

# Get script directory for sourcing
script_dir <- if (interactive()) {
  dirname(rstudioapi::getActiveDocumentContext()$path)
} else {
  dirname(sys.frame(1)$ofile)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

run_narps <- TRUE
run_sheng2020 <- TRUE
run_combine <- TRUE
run_exclusions <- TRUE
run_summary <- TRUE

if (length(args) > 0) {
  run_narps <- "narps" %in% args || "all" %in% args
  run_sheng2020 <- "sheng2020" %in% args || "all" %in% args
  run_combine <- "combine" %in% args || "all" %in% args
  run_exclusions <- "exclusions" %in% args || "all" %in% args
  run_summary <- "summary" %in% args || "all" %in% args

  # If only specific steps requested, adjust defaults
  if (!("all" %in% args)) {
    if (!("narps" %in% args)) run_narps <- FALSE
    if (!("sheng2020" %in% args)) run_sheng2020 <- FALSE
    if (!("combine" %in% args)) run_combine <- FALSE
    if (!("exclusions" %in% args)) run_exclusions <- FALSE
    if (!("summary" %in% args)) run_summary <- FALSE
  }
}

cat("=============================================================================\n")
cat("Pupil Data Preprocessing Pipeline v2\n")
cat("=============================================================================\n\n")

# =============================================================================
# Step 1: NARPS Preprocessing
# =============================================================================

if (run_narps) {
  cat(">>> Step 1: NARPS Preprocessing\n")
  cat("-----------------------------------------------------------------------------\n")
  tryCatch({
    source(file.path(script_dir, "1_preprocess_narps.R"))
  }, error = function(e) {
    cat("ERROR in NARPS preprocessing:\n")
    cat(conditionMessage(e), "\n")
    if (!interactive()) quit(status = 1)
  })
  cat("\n")
} else {
  cat(">>> Step 1: NARPS Preprocessing [SKIPPED]\n\n")
}

# =============================================================================
# Step 2: Sheng2020 Preprocessing
# =============================================================================

if (run_sheng2020) {
  cat(">>> Step 2: Sheng2020 Preprocessing\n")
  cat("-----------------------------------------------------------------------------\n")
  tryCatch({
    source(file.path(script_dir, "1_preprocess_sheng2020.R"))
  }, error = function(e) {
    cat("ERROR in Sheng2020 preprocessing:\n")
    cat(conditionMessage(e), "\n")
    if (!interactive()) quit(status = 1)
  })
  cat("\n")
} else {
  cat(">>> Step 2: Sheng2020 Preprocessing [SKIPPED]\n\n")
}

# =============================================================================
# Step 3: Combine Datasets
# =============================================================================

if (run_combine) {
  cat(">>> Step 3: Combine Datasets\n")
  cat("-----------------------------------------------------------------------------\n")
  tryCatch({
    source(file.path(script_dir, "2_combine_datasets.R"))
  }, error = function(e) {
    cat("ERROR in combining datasets:\n")
    cat(conditionMessage(e), "\n")
    if (!interactive()) quit(status = 1)
  })
  cat("\n")
} else {
  cat(">>> Step 3: Combine Datasets [SKIPPED]\n\n")
}

# =============================================================================
# Step 4: Apply Exclusions
# =============================================================================

if (run_exclusions) {
  cat(">>> Step 4: Apply Exclusions\n")
  cat("-----------------------------------------------------------------------------\n")
  tryCatch({
    source(file.path(script_dir, "3_exclusions.R"))
  }, error = function(e) {
    cat("ERROR in applying exclusions:\n")
    cat(conditionMessage(e), "\n")
    if (!interactive()) quit(status = 1)
  })
  cat("\n")
} else {
  cat(">>> Step 4: Apply Exclusions [SKIPPED]\n\n")
}

# =============================================================================
# Step 5: Summary Statistics
# =============================================================================

if (run_summary) {
  cat(">>> Step 5: Summary Statistics\n")
  cat("-----------------------------------------------------\n")
  tryCatch({
    source(file.path(script_dir, "4_summary_statistics.R"))
  }, error = function(e) {
    cat("ERROR in summary statistics:\n")
    cat(conditionMessage(e), "\n")
    if (!interactive()) quit(status = 1)
  })
  cat("\n")
} else {
  cat(">>> Step 5: Summary Statistics [SKIPPED]\n\n")
}

# =============================================================================
# Done
# =============================================================================

cat("=============================================================================\n")
cat("Pipeline Complete!\n")
cat("=============================================================================\n\n")

# Print output file locations
source(file.path(script_dir, "0_config.R"))
cat("Output files:\n")
cat("  - ", output_start_aligned, "\n")
cat("  - ", output_response_aligned, "\n")
cat("  - ", output_statistics, "\n")
cat("  - ", file.path(results_path, "setup_pupildata",
                      "pupil_data_statistics_with_exclusions.csv"), "\n")
cat("  - ", file.path(results_path, "setup_pupildata",
                      "pupil_subject_exclusions.csv"), "\n")
cat("  - ", output_summary_statistics, "\n")
