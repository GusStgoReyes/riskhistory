# =============================================================================
# Choice History Under Risk: Pupil Timeseries — Timepoint LME
# =============================================================================
# Fits one LME per timebin per dataset for both start- and response-aligned
# epochs. The accept and switch fixed effects (estimate, SE, t-stat, p-value)
# across timebins are the input for downstream cluster-permutation analysis.
# =============================================================================

pupillme_path <- file.path(repo_path, "src", "riskhistory", "analysis_pupillme")
source(file.path(pupillme_path, "utils_modelanalytics.R"))
source(file.path(pupillme_path, "utils_timepointmodels.R"))

formula_m1 <- pupil_size ~ logRT_z + gain_z + loss_z + entropy_c +
  log_abs_delta_sv_z + accept + switch + accept:switch + (1 | sub)

# =============================================================================
# Start-aligned
# =============================================================================
message("Fitting start-aligned timepoint models...")
results_start <- run_all_timepoint_models(
  data      = data_start_clean,
  formula   = formula_m1,
  datasets  = datasets,
  p_method  = p_method
)

output_start <- results_start$fixed_effects %>%
  select(timebin, dataset, term, estimate, std.error, statistic,
         conf.low, conf.high, p.value, p_adj_bh, converged, singular)

write.csv(output_start,
          file.path(analysis_results_path,
                    "1_timeseries_start_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_start_fixed_effects.csv")

# =============================================================================
# Response-aligned
# =============================================================================
message("Fitting response-aligned timepoint models...")
results_response <- run_all_timepoint_models(
  data      = data_response_clean,
  formula   = formula_m1,
  datasets  = datasets,
  p_method  = p_method
)

output_response <- results_response$fixed_effects %>%
  select(timebin, dataset, term, estimate, std.error, statistic,
         conf.low, conf.high, p.value, p_adj_bh, converged, singular)

write.csv(output_response,
          file.path(analysis_results_path,
                    "1_timeseries_response_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_fixed_effects.csv")
