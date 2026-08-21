# =============================================================================
# Choice History Under Risk: Pupil Timeseries — Timepoint LME
# =============================================================================

source(file.path(acceptglme_path, "utils_modelanalytics.R"))
source(file.path(utils_path, "utils_timepointmodels.R"))

formula_m1 <- pupil_size ~ logRT_z + gain_z + loss_z + entropy_c +
  abs_delta_ev_z + transition + (1 | sub)


TRANSITION_PAIRS <- list(
  c("accept_to_reject", "reject_to_reject"),
  c("reject_to_accept", "accept_to_accept")
)

# =============================================================================
# Start-aligned
# =============================================================================
message("Fitting start-aligned timepoint models...")
results_start <- run_all_timepoint_models(
  data           = data_start_clean,
  formula        = formula_m1,
  datasets       = datasets,
  p_method       = p_method,
  emmeans_var    = "transition",
  contrast_pairs = TRANSITION_PAIRS
)

output_start <- results_start$fixed_effects %>%
  select(timebin, dataset, term, estimate, std.error, statistic,
         conf.low, conf.high, p.value, p_adj_bh, converged, singular)

write.csv(output_start,
          file.path(analysis_results_path,
                    "1_timeseries_start_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_start_fixed_effects.csv")

write.csv(results_start$avgpred,
          file.path(analysis_results_path,
                    "1_timeseries_start_avgpred.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_start_avgpred.csv")

write.csv(results_start$contrasts,
          file.path(analysis_results_path,
                    "1_timeseries_start_contrasts.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_start_contrasts.csv")

# =============================================================================
# Response-aligned
# =============================================================================
message("Fitting response-aligned timepoint models...")
results_response <- run_all_timepoint_models(
  data           = data_response_clean,
  formula        = formula_m1,
  datasets       = datasets,
  p_method       = p_method,
  emmeans_var    = "transition",
  contrast_pairs = TRANSITION_PAIRS
)

output_response <- results_response$fixed_effects %>%
  select(timebin, dataset, term, estimate, std.error, statistic,
         conf.low, conf.high, p.value, p_adj_bh, converged, singular)

write.csv(output_response,
          file.path(analysis_results_path,
                    "1_timeseries_response_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_fixed_effects.csv")

write.csv(results_response$avgpred,
          file.path(analysis_results_path,
                    "1_timeseries_response_avgpred.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_avgpred.csv")

write.csv(results_response$contrasts,
          file.path(analysis_results_path,
                    "1_timeseries_response_contrasts.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_contrasts.csv")
