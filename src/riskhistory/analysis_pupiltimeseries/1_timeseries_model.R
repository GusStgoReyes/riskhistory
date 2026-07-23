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

# Three grouped contrasts, each averaging 2-vs-2 or 1-vs-1 transition levels
# via marginaleffects::hypotheses() (see utils_timepointmodels.R):
#   1. Repeat vs Switch: (accept_to_accept + reject_to_reject)/2 vs
#      (accept_to_reject + reject_to_accept)/2 -- did pupil response differ
#      between staying and switching, collapsing over previous-choice
#      outcome.
#   2. Switch to Accept vs Switch to Reject: reject_to_accept vs
#      accept_to_reject directly -- of the two ways to switch, did pupil
#      response differ by which direction the switch went.
#   3. Repeat Accept vs Repeat Reject: accept_to_accept vs reject_to_reject
#      directly -- of the two ways to stay, did pupil response differ by
#      which outcome was repeated.
# Passed to run_all_timepoint_models() so emmeans/contrasts are computed per
# timepoint off the already-fit model (no refit), with p_adj_bh on the
# contrasts table adjusted per dataset across every timepoint and all three
# contrasts together (a family separate from the fixed-effects p_adj_bh).
TRANSITION_CONTRASTS <- list(
  list(
    name = "Repeat - Switch",
    pos = c("accept_to_accept", "reject_to_reject"), pos_label = "Repeat",
    neg = c("accept_to_reject", "reject_to_accept"), neg_label = "Switch"
  ),
  list(
    name = "Switch to Accept - Switch to Reject",
    pos = "reject_to_accept", pos_label = "Switch to Accept",
    neg = "accept_to_reject", neg_label = "Switch to Reject"
  ),
  list(
    name = "Repeat Accept - Repeat Reject",
    pos = "accept_to_accept", pos_label = "Repeat Accept",
    neg = "reject_to_reject", neg_label = "Repeat Reject"
  )
)

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
