# =============================================================================
# Choice History Under Risk: Pupil LME Analysis - Start Aligned
# =============================================================================
# This script runs linear mixed effects models on pupil dilation data
# aligned to stimulus onset (start time), examining the effect of previous
# choice on pupil dilation for each dataset separately.
#
# Output: CSV with columns: timebin, dataset, term, estimate, conf.low,
#         conf.high, converged, p_adj_bh
# =============================================================================

# Source the config file (loads data and packages)
source(file.path(dirname(sys.frame(1)$ofile), "0_config.R"))

# =============================================================================
# Run LME models for each timebin and dataset
# =============================================================================
message("Running start-aligned pupil models...")

# Initialize results dataframe
results_all <- data.frame()

# Loop through datasets
for (ds in datasets) {
  message(sprintf("  Processing dataset: %s", ds))

  # Get data for this dataset
  ds_data <- data_start_clean %>%
    filter(dataset == ds)

  # Check if we have data
  if (nrow(ds_data) == 0) {
    warning(sprintf("No data for dataset %s, skipping...", ds))
    next
  }

  # Get timebins for this dataset
  ds_timebins <- sort(unique(ds_data$timebin))
  message(sprintf("    %d timebins, %d observations", length(ds_timebins), nrow(ds_data)))

  # Loop through timebins
  for (tb in ds_timebins) {
    # Get data for this timebin
    tb_data <- ds_data %>%
      filter(timebin == tb)

    # Skip if insufficient data
    if (nrow(tb_data) < 50) {
      warning(sprintf("Insufficient data for timebin %.3f in %s, skipping...", tb, ds))
      next
    }

    # Fit the model
    # Model: pupil_size ~ sv + entropy + logRT + choice_prev1 + switch + random effects
    model <- tryCatch({
      lmer(pupil_size ~ sv + entropy + logRT + choice_prev1 + switch +
             (1 | sub) +
             (-1 + sv | sub) +
             (-1 + entropy | sub) +
             (-1 + logRT | sub),
           data = tb_data,
           control = lmerControl(optimizer = "bobyqa",
                                 optCtrl = list(maxfun = 100000)))
    }, error = function(e) {
      warning(sprintf("Model failed for timebin %.3f in %s: %s", tb, ds, e$message))
      return(NULL)
    })

    # Skip if model failed
    if (is.null(model)) next

    # Check convergence
    converged <- check_model_convergence(model)

    # Extract fixed effects with confidence intervals
    model_results <- tryCatch({
      tidy(model, effects = "fixed", conf.int = TRUE, conf.level = 0.95)
    }, error = function(e) {
      warning(sprintf("Failed to extract results for timebin %.3f in %s: %s", tb, ds, e$message))
      return(NULL)
    })

    if (is.null(model_results)) next

    # Add metadata
    model_results$timebin <- tb
    model_results$dataset <- ds
    model_results$converged <- converged

    # Append to results
    results_all <- rbind(results_all, model_results)
  }
}

# =============================================================================
# Apply multiple comparisons correction
# =============================================================================
message("Applying Benjamini-Hochberg correction...")

# Apply BH correction across all p-values
results_all <- results_all %>%
  mutate(p_adj_bh = p.adjust(p.value, method = p_method))

# =============================================================================
# Select and rename output columns
# =============================================================================
results_output <- results_all %>%
  select(timebin, dataset, term, estimate, conf.low, conf.high, converged, p_adj_bh)

# =============================================================================
# Save results
# =============================================================================
output_file <- file.path(analysis_results_path, "1_pupilmodel_startAligned.csv")
write.csv(results_output, output_file, row.names = FALSE)
message(sprintf("Results saved to: %s", output_file))

# Print summary
message("\n=== Summary ===")
message(sprintf("Total results: %d rows", nrow(results_output)))
for (ds in datasets) {
  ds_results <- results_output %>% filter(dataset == ds)
  n_timebins <- length(unique(ds_results$timebin))
  n_converged <- sum(ds_results$converged) / length(unique(ds_results$term))
  message(sprintf("  %s: %d timebins, %.0f%% converged",
                  ds, n_timebins, 100 * n_converged / n_timebins))
}

# Print significant effects for choice_prev1
message("\n=== Significant choice_prev1 effects (p_adj_bh < 0.05) ===")
sig_choice <- results_output %>%
  filter(term == "choice_prev1" & p_adj_bh < 0.05)
if (nrow(sig_choice) > 0) {
  print(sig_choice)
} else {
  message("No significant effects found.")
}

message("\nStart-aligned analysis complete.")
