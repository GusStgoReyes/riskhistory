# =============================================================================
# Pupil Betas Incremental Model Building
# =============================================================================

model_names <- c(
  "M0_RT", "M1_values", "M2_uncertainty", "M3_surprise",
  "M4_risk", "M5_switch", "M6_entrenchment", "M7_alternation"
)

# =============================================================================
# Fit models per dataset (REML = FALSE for LRT / AIC comparison)
# =============================================================================
models_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting models for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  M0_RT <- lmer(
    pupil ~ logRT_z + (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  M1_values <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  M2_uncertainty <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + entropy_c + (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  M3_surprise <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  M4_risk <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + accept + (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  M5_switch <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + accept + switch + (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  M6_switch_accept <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + entropy_c +
      log_abs_delta_sv_z + accept * switch +
      (1 + logRT_z | sub),
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  models_list[[ds]] <- list(
    M0_RT           = M0_RT,
    M1_values       = M1_values,
    M2_uncertainty  = M2_uncertainty,
    M3_surprise     = M3_surprise,
    M4_risk         = M4_risk,
    M5_switch       = M5_switch,
    M6_switch_accept = M6_switch_accept
  )
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(models_list, datasets, model_names)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "1_pupilbetas_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbetas_convergence_report.csv")

# =============================================================================
# Sequential LRT (anova)
# =============================================================================
lrt_results <- run_anova_comparison(models_list, datasets, model_names)

write.csv(lrt_results,
          file.path(analysis_results_path,
                    "1_pupilbetas_lrt.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbetas_lrt.csv")

message("\nIncremental model building complete.")
