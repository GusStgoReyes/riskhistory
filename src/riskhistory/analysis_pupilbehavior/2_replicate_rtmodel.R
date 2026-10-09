# =============================================================================
# Pupil-Behavior — Replicate RT Model
# =============================================================================

replicatertmodel_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting replicate RT model for dataset: %s", ds))
  df_sub <- data_pupilbehavior %>% filter(dataset == ds)

  m_switch <- glmer(
    rt ~ sv_abs_z + switch + (1 + sv_abs_z + switch | sub),
    data = df_sub,
    family = Gamma(link = "identity"),
    control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m_switch)
  message(sprintf("  converged: %s, singular: %s",
                  diag$converged, diag$singular))

  replicatertmodel_list[[ds]] <- list(replicate_rt_switch = m_switch)
}

# =============================================================================
# Fixed effects
# =============================================================================
message("\nExtracting fixed effects...")
replicatertmodel_fixed_effects <- extract_fixed_effects(
  model_list = replicatertmodel_list,
  datasets   = datasets,
  models     = c("replicate_rt_switch"),
  p_method   = p_method
)
write.csv(replicatertmodel_fixed_effects,
          file.path(analysis_results_path,
                    "2_replicate_rtmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_replicate_rtmodel_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  replicatertmodel_list, datasets, c("replicate_rt_switch")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "2_replicate_rtmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 2_replicate_rtmodel_convergence.csv")

message("\nReplicate RT model analysis complete.")
