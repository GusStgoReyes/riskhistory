# =============================================================================
# Pupil-Behavior — Accept Model
# =============================================================================
# accept ~ gain_centered + loss_centered + choice_prev1 * pupil +
#   (1 + gain_centered + loss_centered | sub)

formula_pupilbehavior_accept <- accept ~ gain_centered + loss_centered +
  choice_prev1 * pupil +
  (1 + gain_centered + loss_centered | sub)

acceptmodel_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting accept model for dataset: %s", ds))
  df <- data_pupilbehavior %>% filter(dataset == ds)

  m <- glmer(
    formula_pupilbehavior_accept,
    data = df, family = binomial, control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  acceptmodel_list[[ds]] <- list(pupilbehavior_accept = m)
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  acceptmodel_list, datasets, c("pupilbehavior_accept")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "1_pupilbehavior_acceptmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_acceptmodel_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = acceptmodel_list,
  datasets   = datasets,
  models     = "pupilbehavior_accept",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "1_pupilbehavior_acceptmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_acceptmodel_fixed_effects.csv")

message("\nPupil-behavior accept model analysis complete.")
