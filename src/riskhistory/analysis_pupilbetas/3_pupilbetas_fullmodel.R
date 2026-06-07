# =============================================================================
# Pupil Betas — M5 with Maximal Random Effects
# =============================================================================

formulas_m5_full <- list(
  # Entropy uses BLUPS and is a function of gain and loss
  # Instead of the BLUPS use the per subject model.
  # Proceed with caution for the regressors w combinations of gain and loss
  equalIndifference = pupil ~
    logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + accept + switch + accept:switch +
      (1 + logRT_z + gain_z + entropy_c +
         log_abs_delta_sv_z + accept + accept:switch || sub),

  equalRange = pupil ~
    logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + accept + switch + accept:switch +
      (1 + logRT_z + loss_z + entropy_c +
         log_abs_delta_sv_z + accept + switch || sub),

  sheng2020 = pupil ~
    logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + accept + switch + accept:switch +
      (1 + logRT_z + gain_z + loss_z +
         + accept + switch + accept:switch || sub)
)

# =============================================================================
# Fit per dataset (REML = TRUE — this is the final reporting model)
# =============================================================================
fullmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting full model for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  m5_full <- lmer(formulas_m5_full[[ds]],
                  data = df, REML = TRUE, control = LMER_CONTROL)
  print(summary(m5_full)$varcor)

  fullmodel_list[[ds]] <- list(M5_full = m5_full)
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(fullmodel_list, datasets, "M5_full")

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "3_pupilbetas_fullmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_fullmodel_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = fullmodel_list,
  datasets   = datasets,
  models     = "M5_full",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "3_pupilbetas_fullmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_fullmodel_fixed_effects.csv")

# =============================================================================
# Per-subject BLUPs for accept and switch
# =============================================================================
blups <- extract_blups(fullmodel_list, datasets, "M5_full")

write.csv(blups,
          file.path(analysis_results_path,
                    "3_pupilbetas_fullmodel_blups.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_fullmodel_blups.csv")
