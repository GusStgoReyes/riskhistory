# =============================================================================
# Choice History Under Risk: Pupil Timeseries — Timepoint LME
# =============================================================================
# Fits one LME per timebin per dataset for both start- and response-aligned
# epochs. The transition fixed effect (estimate, SE, t-stat, p-value) across
# timebins is the input for downstream cluster-permutation analysis.
#
# `transition` is a 4-level factor (accept_to_accept, accept_to_reject,
# reject_to_accept, reject_to_reject; reference = accept_to_accept)
# =============================================================================

source(file.path(acceptglme_path, "utils_modelanalytics.R"))
source(file.path(utils_path, "utils_timepointmodels.R"))

formula_m1 <- pupil_size ~ logRT_z + gain_z + loss_z + entropy_c +
  abs_delta_ev_z + transition + (1 | sub)


TRANSITION_LEVELS <- c("accept_to_accept", "accept_to_reject",
                       "reject_to_accept", "reject_to_reject")
TRANSITION_PAIRS <- combn(TRANSITION_LEVELS, 2, simplify = FALSE)
TRANSITION_CONTRASTS <- lapply(TRANSITION_PAIRS, function(pair) {
  list(
    name = paste(pair[1], "-", pair[2]),
    pos = pair[1], pos_label = pair[1],
    neg = pair[2], neg_label = pair[2]
  )
})

# =============================================================================
# Start-aligned
# =============================================================================
message("Fitting start-aligned timepoint models...")
results_start <- run_all_timepoint_models(
  data              = data_start_clean,
  formula           = formula_m1,
  datasets          = datasets,
  p_method          = p_method,
  emmeans_var       = "transition",
  emmeans_contrasts = TRANSITION_CONTRASTS
)

output_start <- results_start$fixed_effects %>%
  select(timebin, dataset, term, estimate, std.error, statistic,
         conf.low, conf.high, p.value, p_adj_bh, converged, singular)

write.csv(output_start,
          file.path(analysis_results_path,
                    "1_timeseries_start_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_start_fixed_effects.csv")

write.csv(results_start$emmeans,
          file.path(analysis_results_path,
                    "1_timeseries_start_emmeans.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_start_emmeans.csv")

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
  data              = data_response_clean,
  formula           = formula_m1,
  datasets          = datasets,
  p_method          = p_method,
  emmeans_var       = "transition",
  emmeans_contrasts = TRANSITION_CONTRASTS
)

output_response <- results_response$fixed_effects %>%
  select(timebin, dataset, term, estimate, std.error, statistic,
         conf.low, conf.high, p.value, p_adj_bh, converged, singular)

write.csv(output_response,
          file.path(analysis_results_path,
                    "1_timeseries_response_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_fixed_effects.csv")

write.csv(results_response$emmeans,
          file.path(analysis_results_path,
                    "1_timeseries_response_emmeans.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_emmeans.csv")

write.csv(results_response$contrasts,
          file.path(analysis_results_path,
                    "1_timeseries_response_contrasts.csv"),
          row.names = FALSE)
message("Saved: 1_timeseries_response_contrasts.csv")
