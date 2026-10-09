# =============================================================================
# Pupil Timeseries — one LME per timebin and dataset, per alignment
# =============================================================================

formula_m1 <- pupil_size ~ rt_z + gain_z + loss_z + entropy_z +
  abs_delta_ev_z + transition + (1 | sub)

TRANSITION_PAIRS <- list(
  c("accept_to_reject", "reject_to_reject"),
  c("reject_to_accept", "accept_to_accept")
)

# Equal-weight pooled means by current choice; contrast = accept - reject
POOLED_TRANSITIONS <- list(
  accept = c("accept_to_accept", "reject_to_accept"),
  reject = c("reject_to_reject", "accept_to_reject")
)

# Alignments to fit; remove one to run only the other
ALIGNMENTS <- list(
  start    = data_start_clean,
  response = data_response_clean
)

FIXED_EFFECT_COLS <- c("timebin", "dataset", "term", "estimate", "std.error",
                       "statistic", "conf.low", "conf.high", "p.value",
                       "p_adj_bh", "converged", "singular")

save_csv <- function(df, filename) {
  write.csv(df, file.path(analysis_results_path, filename), row.names = FALSE)
  message("Saved: ", filename)
}

# =============================================================================
# Fit and save
# =============================================================================
for (alignment in names(ALIGNMENTS)) {
  message(sprintf("\nFitting %s-aligned timepoint models...", alignment))
  results <- run_all_timepoint_models(
    data           = ALIGNMENTS[[alignment]],
    formula        = formula_m1,
    datasets       = datasets,
    p_method       = p_method,
    emmeans_var    = "transition",
    contrast_pairs = TRANSITION_PAIRS,
    pooled_levels  = POOLED_TRANSITIONS
  )

  prefix <- paste0("1_timeseries_", alignment)
  save_csv(select(results$fixed_effects, all_of(FIXED_EFFECT_COLS)),
           paste0(prefix, "_fixed_effects.csv"))
  save_csv(results$avgpred,          paste0(prefix, "_avgpred.csv"))
  save_csv(results$contrasts,        paste0(prefix, "_contrasts.csv"))
  save_csv(results$avgpred_choice,   paste0(prefix, "_avgpred_choice.csv"))
  save_csv(results$contrasts_choice, paste0(prefix, "_contrasts_choice.csv"))
}
