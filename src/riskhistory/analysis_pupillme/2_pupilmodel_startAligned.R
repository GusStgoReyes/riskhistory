# =============================================================================
# Choice History Under Risk: Pupil LME Analysis - Start Aligned
# =============================================================================
# This script runs linear mixed effects models on pupil dilation data
# aligned to stimulus onset (start time), examining the effect of previous
# choice on pupil dilation for each dataset separately.
#
# P-value adjustment (BH) is applied within each dataset across all
# estimates and timebins.
#
# Output: CSV with columns: timebin, dataset, term, estimate, conf.low,
#         conf.high, converged, p_adj_bh
# =============================================================================

# =============================================================================
# Define the model formula
# =============================================================================
# Model: pupil_size ~ sv + entropy + logRT + choice + switch + random effects
model_formula <- pupil_size ~ sv + entropy + logRT + choice * switch + (1 + switch || sub)


# =============================================================================
# Run LME models for each timebin and dataset
# =============================================================================
message("Running start-aligned pupil models...")

results_all <- run_all_timepoint_models(
  data = data_start_clean,
  formula = model_formula,
  datasets = datasets,
  p_method = p_method
)

# =============================================================================
# Select and rename output columns
# =============================================================================
results_output <- results_all$fixed_effects %>%
  select(timebin, dataset, term, estimate, conf.low, conf.high,
         converged, p_adj_bh)

# =============================================================================
# Save results
# =============================================================================
output_file <- file.path(analysis_results_path, "2_pupilmodel_startAligned.csv")
write.csv(results_output, output_file, row.names = FALSE)
message(sprintf("Results saved to: %s", output_file))

# =============================================================================
# Save random effects for switch
# =============================================================================
if (!is.null(results_all$random_effects)) {
  random_effects_file <- file.path(analysis_results_path, "3_randomeffect_switch_startAligned.csv")
  write.csv(results_all$random_effects, random_effects_file, row.names = FALSE)
  message(sprintf("Random effects saved to: %s", random_effects_file))
}

# =============================================================================
# Print summary
# =============================================================================
print_results_summary(results_output, datasets)
print_significant_effects(results_output, "switch")

message("\nStart-aligned analysis complete.")
