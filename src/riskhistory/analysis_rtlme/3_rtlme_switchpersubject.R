# =============================================================================
# Per-subject switch effect on RT (Gamma GLM per subject, no partial pooling)
# =============================================================================

# =============================================================================
# Fit per subject
# =============================================================================
persubject_switch_betas <- bind_rows(lapply(datasets, function(ds) {
  message(sprintf("Fitting per-subject switch-effect RT models for dataset: %s",
                  ds))
  df_ds <- rtlme_data[rtlme_data$dataset == ds, ]

  bind_rows(lapply(sort(unique(df_ds$sub)), function(s) {
    fit <- fit_switch_glm(df_ds[df_ds$sub == s, ])
    data.frame(
      dataset       = ds,
      sub           = s,
      switch_effect = fit$effect,
      converged     = fit$converged
    )
  }))
}))

message(sprintf("Per-subject switch-effect RT models: %d of %d converged.",
                sum(persubject_switch_betas$converged),
                nrow(persubject_switch_betas)))

# =============================================================================
# Save
# =============================================================================
betas_file <- "3_rtlme_switchpersubject_betas.csv"
write.csv(persubject_switch_betas, file.path(analysis_results_path, betas_file),
          row.names = FALSE)
message("Saved: ", betas_file)
