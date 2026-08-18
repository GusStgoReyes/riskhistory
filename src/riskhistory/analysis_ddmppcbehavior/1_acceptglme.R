# =============================================================================
# DDM PPC Behavior: Accept GLME
# =============================================================================
# Per dataset: (1) baseline model accept ~ gain + loss + (1 + gain + loss |
# sub), used only to save per-subject BLUPs; (2) full model adding
# choice_prev1, whose fixed effects are saved.

models_list <- list()
baseline_betas <- list()

for (ds in datasets) {
  message(sprintf("\nFitting DDM PPC accept models for dataset: %s", ds))
  df_sub <- ddm_data[ddm_data$dataset == ds, ]

  # BASELINE MODEL (used only for per-subject betas)
  baseline_model <- glmer(accept ~ gain + loss + (1 + gain + loss | sub),
                          data = df_sub, family = binomial, control = GLMER_CONTROL)
  baseline_betas[[ds]] <- extract_random_effects_baseline(baseline_model, ds)

  # FULL MODEL (choice-history effect)
  full_model <- glmer(accept ~ gain + loss + choice_prev1 + (1 + gain + loss | sub),
                      data = df_sub, family = binomial, control = GLMER_CONTROL)

  models <- list(
    baseline = baseline_model,
    full     = full_model
  )
  models_list <- add_models_to_list(
    models_list = models_list,
    ds     = ds,
    models = models
  )
}

# =============================================================================
# Baseline per-subject betas
# =============================================================================
baseline_betas_df <- bind_rows(baseline_betas)
write.csv(baseline_betas_df,
          file.path(analysis_results_path, "1_acceptglme_baseline_betas.csv"),
          row.names = FALSE)
message("Saved: 1_acceptglme_baseline_betas.csv")

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  models_list, datasets, c("baseline", "full")
)
write.csv(convergence_df,
          file.path(analysis_results_path, "1_acceptglme_convergence.csv"),
          row.names = FALSE)
message("Saved: 1_acceptglme_convergence.csv")

# =============================================================================
# Full-model fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = models_list,
  datasets   = datasets,
  models     = "full",
  p_method   = p_method
)
write.csv(fixed_effects,
          file.path(analysis_results_path, "1_acceptglme_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_acceptglme_fixed_effects.csv")

message("\nDDM PPC accept GLME analysis complete.")
