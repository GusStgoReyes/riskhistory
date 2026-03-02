# =============================================================================
# Choice History Under Risk: Pupil LME Analysis - Two-Back History (Start Aligned)
# =============================================================================
# This script runs linear mixed effects models on pupil dilation data
# aligned to stimulus onset, examining how the two-trial choice history
# (prev2 -> prev1) predicts current pupil size.
#
# Two models are fit per timebin, one for each value of choice_prev1:
#   Model "reject" (choice_prev1 == -1):
#     pupil_size ~ reject_reject + accept_reject + (1 | sub)
#   Model "accept" (choice_prev1 == 1):
#     pupil_size ~ accept_accept + reject_accept + (1 | sub)
#
# P-value adjustment (BH) is applied within each dataset × model combination.
#
# Output: CSV with columns: timebin, dataset, model, term, estimate, conf.low,
#         conf.high, converged, p_adj_bh
# =============================================================================


# =============================================================================
# Define model formulas
# =============================================================================
# For trials where the previous choice was a rejection (choice_prev1 == -1):
#   reject_reject: prev2 reject, prev1 reject (reference: accept_reject = 0)
#   accept_reject: prev2 accept, prev1 reject
formula_reject <- pupil_size ~ reject_reject + accept_reject + (1 | sub)

# For trials where the previous choice was an acceptance (choice_prev1 == 1):
#   accept_accept: prev2 accept, prev1 accept (reference: reject_accept = 0)
#   reject_accept: prev2 reject, prev1 accept
formula_accept <- pupil_size ~ accept_accept + reject_accept + (1 | sub)


# =============================================================================
# Split data by choice_prev1
# =============================================================================
data_reject_subset <- data_start_clean %>%
  filter(choice_prev1 == -1)

data_accept_subset <- data_start_clean %>%
  filter(choice_prev1 == 1)

message(sprintf("Reject subset: %d rows", nrow(data_reject_subset)))
message(sprintf("Accept subset: %d rows", nrow(data_accept_subset)))


# =============================================================================
# Run models for reject subset (choice_prev1 == -1)
# P-value adjustment applied within each dataset (one model per dataset)
# =============================================================================
message("Running two-back models for reject subset (choice_prev1 == -1)...")

results_reject <- run_all_timepoint_models(
  data = data_reject_subset,
  formula = formula_reject,
  datasets = datasets,
  p_method = p_method
)

results_reject$fixed_effects <- results_reject$fixed_effects %>%
  mutate(model = "reject")


# =============================================================================
# Run models for accept subset (choice_prev1 == 1)
# P-value adjustment applied within each dataset (one model per dataset)
# =============================================================================
message("Running two-back models for accept subset (choice_prev1 == 1)...")

results_accept <- run_all_timepoint_models(
  data = data_accept_subset,
  formula = formula_accept,
  datasets = datasets,
  p_method = p_method
)

results_accept$fixed_effects <- results_accept$fixed_effects %>%
  mutate(model = "accept")


# =============================================================================
# Combine results and select output columns
# =============================================================================
results_combined <- bind_rows(results_reject$fixed_effects,
                              results_accept$fixed_effects)

results_output <- results_combined %>%
  select(timebin, dataset, model, term, estimate, conf.low, conf.high,
         converged, p_adj_bh)


# =============================================================================
# Save results
# =============================================================================
output_file <- file.path(analysis_results_path, "3_pupilmodel_switch.csv")
write.csv(results_output, output_file, row.names = FALSE)
message(sprintf("Results saved to: %s", output_file))


# =============================================================================
# Print summary
# =============================================================================
message("\n--- Reject subset summary ---")
print_results_summary(results_reject$fixed_effects, datasets)
print_significant_effects(results_reject$fixed_effects, "reject_reject")
print_significant_effects(results_reject$fixed_effects, "accept_reject")

message("\n--- Accept subset summary ---")
print_results_summary(results_accept$fixed_effects, datasets)
print_significant_effects(results_accept$fixed_effects, "accept_accept")
print_significant_effects(results_accept$fixed_effects, "reject_accept")

message("\nTwo-back switch analysis complete.")
