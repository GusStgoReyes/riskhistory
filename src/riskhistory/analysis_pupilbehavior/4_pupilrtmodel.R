# =============================================================================
# Pupil-Behavior — RT Model
# =============================================================================
# log(rt) ~ sv_abs + switch * pupil + (1 + sv_abs | sub), fit separately on
# accept == 1 trials and accept == 0 trials (per dataset), since a switch
# following an accept and a switch following a reject are qualitatively
# different events.

formula_pupilbehavior_rt <- log(rt) ~ sv_abs * pupil + switch * pupil + (1 + sv_abs | sub)

RT_MODEL_NAMES <- c(
  accept = "pupilbehavior_rt_accept",
  reject = "pupilbehavior_rt_reject"
)

rtmodel_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT models for dataset: %s", ds))
  df_ds <- data_pupilbehavior %>% filter(dataset == ds)

  ds_models <- list()
  for (acc in c(1, 0)) {
    label <- if (acc == 1) "accept" else "reject"
    message(sprintf("  Fitting %s-trial model", label))

    df <- df_ds %>% filter(accept == acc)
    m <- lmer(formula_pupilbehavior_rt, data = df, control = LMER_CONTROL)
    diag <- check_model_diagnostics(m)
    message(sprintf("    converged: %s, singular: %s", diag$converged, diag$singular))

    ds_models[[RT_MODEL_NAMES[[label]]]] <- m
  }
  rtmodel_list[[ds]] <- ds_models
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  rtmodel_list, datasets, RT_MODEL_NAMES
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "4_pupilrtmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 4_pupilrtmodel_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = rtmodel_list,
  datasets   = datasets,
  models     = RT_MODEL_NAMES,
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "4_pupilrtmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 4_pupilrtmodel_fixed_effects.csv")

message("\nPupil-behavior RT model analysis complete.")
