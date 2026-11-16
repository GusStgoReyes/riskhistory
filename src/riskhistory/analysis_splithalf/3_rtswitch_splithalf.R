# =============================================================================
# Per-subject switch effect on RT within each half
# =============================================================================

# =============================================================================
# Fit per subject and half
# =============================================================================
splithalf_switch_betas <- bind_rows(lapply(datasets, function(ds) {
  message(sprintf("Fitting split-half switch-effect RT models for dataset: %s", ds))
  df_ds <- splithalf_data[splithalf_data$dataset == ds, ]

  bind_rows(lapply(split(df_ds, list(df_ds$sub, df_ds$half_id), drop = TRUE), function(df_sub) {
    fit <- fit_switch_glm(df_sub)
    data.frame(
      dataset       = ds,
      sub           = df_sub$sub[1],
      half_id       = df_sub$half_id[1],
      switch_effect = fit$effect,
      converged     = fit$converged
    )
  }))
})) %>%
  arrange(dataset, sub, half_id)

message(sprintf("Split-half switch-effect RT models: %d of %d converged.",
                sum(splithalf_switch_betas$converged),
                nrow(splithalf_switch_betas)))

# =============================================================================
# Save
# =============================================================================
betas_file <- "3_rtswitch_splithalf_betas.csv"
write.csv(splithalf_switch_betas, file.path(analysis_results_path, betas_file),
          row.names = FALSE)
message("Saved: ", betas_file)
