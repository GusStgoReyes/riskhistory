# =============================================================================
# Choice History Under Risk: Pupil LME Analysis - Start Aligned
# =============================================================================

# =============================================================================
# Model M1: Visualizing the effects of switching on streak, including controls
# =============================================================================
message("Training M1 (switching + streak)...")
formula_m1 <- pupil_size ~ sv_z + log_abs_delta_sv_z + entropy_c + logRT_z +
  + accept + logstreak_on_switch_z + logstreak_on_repeat_z +
  (1 + logstreak_on_switch_z + accept || sub)

# Start-aligned model M1
results_m1 <- run_all_timepoint_models(
  data = data_start_clean,
  formula = formula_m1,
  datasets = datasets,
  p_method = p_method
)
output_m1 <- results_m1$fixed_effects %>%
  select(timebin, dataset, term, estimate, conf.low, conf.high,
         converged, singular, p_adj_bh)

write.csv(output_m1,
          file.path(analysis_results_path,
                    "2_pupilmodel_startAligned_M1_switchStreak.csv"),
          row.names = FALSE)

# Response aligned model M1
results_m1 <- run_all_timepoint_models(
  data = data_response_clean,
  formula = formula_m1,
  datasets = datasets,
  p_method = p_method
)
output_m1 <- results_m1$fixed_effects %>%
  select(timebin, dataset, term, estimate, conf.low, conf.high,
         converged, singular, p_adj_bh)

write.csv(output_m1,
          file.path(analysis_results_path,
                    "2_pupilmodel_responseAligned_M1_switchStreak.csv"),
          row.names = FALSE)

message("M1 results saved.")

# =============================================================================
# ADD ANY ADDITIONAL MODELS BELOW
# =============================================================================