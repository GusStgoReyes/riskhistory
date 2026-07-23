# =============================================================================
# Choice History Under Risk: Pupil x Choice-History Interaction
# =============================================================================
# Continuous model -- this is the one reported in a table. Residualized pupil
# only moderates choice_prev1 (gain/loss are included as covariates but do
# not interact with pupil_residual). Multiple-comparison correction (BH) is
# scoped only to the pupil-related terms (pupil_residual main effect +
# choice_prev1:pupil_residual), since the gain/loss/choice_prev1 main effects
# are already established (see 0_choicehistoryreplication.R) and are not the
# object of inference here.
#
#   accept ~ gain_centered + loss_centered + choice_prev1 * pupil_residual +
#            (1 + gain_centered + loss_centered + pupil_residual | sub)
# =============================================================================

# NOTE: glmer/broom.mixed name interaction terms by the order factors first
# appear across the *whole* formula, not the order written within each `*`
# (e.g. "loss_centered * pupil_residual" here becomes
# "pupil_residual:loss_centered" since pupil_residual already entered the
# model earlier via "gain_centered * pupil_residual"). Match on the
# "pupil_residual" substring rather than a hardcoded term list so this
# doesn't silently miss terms if that ordering changes.
is_pupil_related_term <- function(term) grepl("pupil_residual", term, fixed = TRUE)

pupilinteraction_models_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting pupil interaction model for dataset: %s", ds))
  df_sub <- data_clean %>% filter(dataset == ds)

  m <- glmer(
    accept ~ gain_centered * pupil_residual + loss_centered * pupil_residual + choice_prev1 * pupil_residual +
      (1 + gain_centered + loss_centered | sub),
    data    = df_sub,
    family  = binomial,
    control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  pupilinteraction_models_list[[ds]] <- list(pupilinteraction = m)
}

# =============================================================================
# Fixed effects: all terms saved, but p_adj_bh_pupil is only computed within
# the pupil-related subset (main effect + choice_prev1:pupil_residual) per
# dataset, not across the full term table
# =============================================================================

message("\nExtracting fixed effects...")
all_fixed_effects <- list()
for (ds in datasets) {
  m <- pupilinteraction_models_list[[ds]][["pupilinteraction"]]
  fe <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE) %>%
    mutate(dataset = ds, model = "pupilinteraction")

  fe$p_adj_bh_pupil <- NA_real_
  is_pupil_term <- is_pupil_related_term(fe$term)
  fe$p_adj_bh_pupil[is_pupil_term] <- p.adjust(fe$p.value[is_pupil_term], method = p_method)

  all_fixed_effects[[ds]] <- fe
}
fixed_effects <- bind_rows(all_fixed_effects)

write.csv(fixed_effects,
          file.path(analysis_results_path, "3_pupilinteraction_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilinteraction_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================

convergence_rows <- lapply(datasets, function(ds) {
  d <- check_model_diagnostics(pupilinteraction_models_list[[ds]][["pupilinteraction"]])
  data.frame(dataset          = ds,
             model            = "pupilinteraction",
             converged        = d$converged,
             singular         = d$singular,
             stringsAsFactors = FALSE)
})
convergence_df <- do.call(rbind, convergence_rows)

write.csv(convergence_df,
          file.path(analysis_results_path, "3_pupilinteraction_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 3_pupilinteraction_convergence_report.csv")

message("\nPupil interaction analysis complete.")
