# =============================================================================
# DDM PPC Behavior: Pupil Accept GLME
# =============================================================================
# Per DDM model x dataset:
# logit_mean_accept ~ gain + loss + real_choice_prev1 * pupil +
#   real_choice_prev1 * logrt_z + (1 + gain + loss | sub)
# logrt_z is log(mean_rt), z-scored within sub/dataset (computed per DDM
# model, since each model's simulated RTs are on their own scale).

all_convergence <- list()
all_fixed_effects <- list()

for (ddm_model in PUPIL_DDM_MODELS) {
  message(sprintf("\n=== DDM model: %s ===", ddm_model))
  ddm_data <- load_ddm_ppc_data(ddm_model) %>%
    group_by(sub, dataset) %>%
    mutate(logrt_z = scale(log(mean_rt), center = TRUE, scale = TRUE)[, 1]) %>%
    ungroup()
  pupilaccept_model_name <- sprintf("pupilaccept_%s", ddm_model)

  models_list <- list()

  for (ds in datasets) {
    message(sprintf("\nFitting DDM PPC pupil accept model for dataset: %s", ds))
    df_sub <- ddm_data[ddm_data$dataset == ds, ]

    m <- lmer(
      logit_mean_accept ~ gain + loss + real_choice_prev1 * pupil +
        real_choice_prev1 * logrt_z + (1 + gain + loss | sub),
      data = df_sub, control = LMER_CONTROL
    )
    diag <- check_model_diagnostics(m)
    message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

    models_list <- add_models_to_list(
      models_list = models_list,
      ds     = ds,
      models = setNames(list(m), pupilaccept_model_name)
    )
  }

  all_convergence[[ddm_model]] <- check_convergence_all_models(
    models_list, datasets, c(pupilaccept_model_name)
  ) %>%
    mutate(ddm_model = ddm_model)

  all_fixed_effects[[ddm_model]] <- extract_fixed_effects(
    model_list = models_list,
    datasets   = datasets,
    models     = pupilaccept_model_name,
    p_method   = p_method
  ) %>%
    mutate(ddm_model = ddm_model)
}

# =============================================================================
# Convergence report (all DDM models, one CSV)
# =============================================================================
convergence_df <- bind_rows(all_convergence)
write.csv(convergence_df,
          file.path(analysis_results_path, "3_pupilaccept_convergence.csv"),
          row.names = FALSE)
message("Saved: 3_pupilaccept_convergence.csv")

# =============================================================================
# Fixed effects (all DDM models, one CSV)
# =============================================================================
fixed_effects_df <- bind_rows(all_fixed_effects)
write.csv(fixed_effects_df,
          file.path(analysis_results_path, "3_pupilaccept_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilaccept_fixed_effects.csv")

message("\nDDM PPC pupil accept GLME analysis complete.")
