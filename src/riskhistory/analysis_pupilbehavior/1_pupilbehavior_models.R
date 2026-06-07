# =============================================================================
# Choice History Under Risk: Pupil-Behavior GLME Analysis
# =============================================================================
# Tests whether decision-period pupil dilation interacts with stimulus values
# and previous choices to predict accept/reject decisions.
#
# =============================================================================

pupilbehavior_models_list <- list()
for (ds in datasets) {
  pupilbehavior_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting pupil-behavior model for dataset: %s", ds))
  df_sub <- data_clean %>% filter(dataset == ds)

  m <- glmer(
    accept ~ gain_centered * pupil  + loss_centered * pupil +
      (1 + loss_centered + gain_centered + pupil | sub),
    data    = df_sub,
    family  = binomial,
    control = GLMER_CONTROL
  )

  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  pupilbehavior_models_list[[ds]] <- list(pupilbehavior = m)
}

# =============================================================================
# Fixed effects
# =============================================================================

message("\nExtracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = pupilbehavior_models_list,
  datasets   = datasets,
  models     = "pupilbehavior",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path, "1_pupilbehavior_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================

convergence_rows <- lapply(datasets, function(ds) {
  d <- check_model_diagnostics(pupilbehavior_models_list[[ds]]$pupilbehavior)
  data.frame(dataset = ds, model = "pupilbehavior",
             converged = d$converged, singular = d$singular,
             stringsAsFactors = FALSE)
})
convergence_df <- do.call(rbind, convergence_rows)

write.csv(convergence_df,
          file.path(analysis_results_path, "1_pupilbehavior_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_convergence_report.csv")

message("\nPupil-behavior analysis complete.")
