# =============================================================================
# Pupil-Behavior — RT Model
# =============================================================================
# log(rt) ~ sv_abs + switch_AA_RA * pupil + switch_RR_AR * pupil +
#   (1 + sv_abs | sub)
# switch_AA_RA/switch_RR_AR (0_config.R) split the single switch * pupil
# interaction by switch direction, so switching into accept and switching
# into reject each get their own effect and pupil interaction instead of
# being pooled under one switch term.

formula_pupilbehavior_rt <- log(rt) ~ sv_abs + switch_AA_RA * pupil +
  switch_RR_AR * pupil + (1 + sv_abs | sub)

rtmodel_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT model for dataset: %s", ds))
  df <- data_pupilbehavior %>% filter(dataset == ds)

  m <- lmer(
    formula_pupilbehavior_rt,
    data = df, control = LMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  rtmodel_list[[ds]] <- list(pupilbehavior_rt = m)
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  rtmodel_list, datasets, c("pupilbehavior_rt")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "2_pupilbehavior_rtmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbehavior_rtmodel_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = rtmodel_list,
  datasets   = datasets,
  models     = "pupilbehavior_rt",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "2_pupilbehavior_rtmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbehavior_rtmodel_fixed_effects.csv")

message("\nPupil-behavior RT model analysis complete.")
