# =============================================================================
# Pupil-Behavior — Replicate Accept Model
# =============================================================================

formula_replicate_baseline <- accept ~ gain_centered + loss_centered + (1 + gain_centered + loss_centered | sub)

formula_replicate_prevchoice <- accept ~ gain_centered + loss_centered + choice_prev1 +
  (1 + gain_centered + loss_centered | sub)

replicateacceptmodel_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting replicate accept models for dataset: %s", ds))
  df <- data_pupilbehavior %>% filter(dataset == ds)

  m_baseline <- glmer(
    formula_replicate_baseline,
    data = df, family = binomial, control = GLMER_CONTROL
  )
  diag_baseline <- check_model_diagnostics(m_baseline)
  message(sprintf("  baseline: converged: %s, singular: %s",
                  diag_baseline$converged, diag_baseline$singular))

  m_prevchoice <- glmer(
    formula_replicate_prevchoice,
    data = df, family = binomial, control = GLMER_CONTROL
  )
  diag_prevchoice <- check_model_diagnostics(m_prevchoice)
  message(sprintf("  previous-choice: converged: %s, singular: %s",
                  diag_prevchoice$converged, diag_prevchoice$singular))

  replicateacceptmodel_list[[ds]] <- list(
    replicate_baseline   = m_baseline,
    replicate_prevchoice = m_prevchoice
  )
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  replicateacceptmodel_list, datasets, c("replicate_baseline", "replicate_prevchoice")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "1_replicate_acceptmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 1_replicate_acceptmodel_convergence.csv")

# =============================================================================
# Model comparison: previous-choice vs. baseline
# =============================================================================
comparison_df <- compare_models(
  model_list      = replicateacceptmodel_list,
  datasets        = datasets,
  models          = c("replicate_baseline", "replicate_prevchoice"),
  reference_model = "replicate_baseline"
)

write.csv(comparison_df,
          file.path(analysis_results_path,
                    "1_replicate_acceptmodel_comparison.csv"),
          row.names = FALSE)
message("Saved: 1_replicate_acceptmodel_comparison.csv")

# =============================================================================
# Fixed effects (both models; a "model" column distinguishes them)
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = replicateacceptmodel_list,
  datasets   = datasets,
  models     = c("replicate_baseline", "replicate_prevchoice"),
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "1_replicate_acceptmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_replicate_acceptmodel_fixed_effects.csv")

message("\nReplicate accept model analysis complete.")
