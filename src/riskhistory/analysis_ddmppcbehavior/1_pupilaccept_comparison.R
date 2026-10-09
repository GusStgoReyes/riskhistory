# =============================================================================
# DDM PPC Behavior: Pupil Accept Model Comparison
# =============================================================================

MODEL_PAIRS <- list(
  stp   = c(baseline = "HDDM_Base_Lag1-stp_Pupil",   pupil = "HDDM_Base_Lag1-stp_Pupil_stp"),
  drift = c(baseline = "HDDM_Base_Lag1-drift_Pupil", pupil = "HDDM_Base_Lag1-drift_Pupil_drift")
)

all_convergence <- list()
all_fixed_effects <- list()

for (pair_name in names(MODEL_PAIRS)) {
  pair <- MODEL_PAIRS[[pair_name]]
  message(sprintf("\n=== Model pair: %s (baseline: %s, pupil: %s) ===",
                  pair_name, pair[["baseline"]], pair[["pupil"]]))

  pair_data <- bind_rows(lapply(names(pair), function(mt) {
    load_ddm_ppc_data(pair[[mt]]) %>%
      mutate(model_type = mt, ddm_model = pair[[mt]])
  })) %>%
    mutate(model_type = factor(model_type, levels = c("baseline", "pupil")))

  models_list <- list()

  for (ds in datasets) {
    message(sprintf("\nFitting DDM PPC pupil accept comparison model for dataset: %s", ds))
    df_sub <- pair_data[pair_data$dataset == ds, ]

    m <- lmer(
      logit_mean_accept ~ gain + loss + real_choice_prev1 * pupil * model_type +
        (1 + gain + loss | sub),
      data = df_sub, control = LMER_CONTROL
    )
    diag <- check_model_diagnostics(m)
    message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

    models_list <- add_models_to_list(
      models_list = models_list,
      ds     = ds,
      models = list(pupilaccept_comparison = m)
    )
  }

  all_convergence[[pair_name]] <- check_convergence_all_models(
    models_list, datasets, c("pupilaccept_comparison")
  ) %>%
    mutate(model_pair = pair_name)

  all_fixed_effects[[pair_name]] <- extract_fixed_effects(
    model_list = models_list,
    datasets   = datasets,
    models     = "pupilaccept_comparison",
    p_method   = p_method
  ) %>%
    mutate(model_pair = pair_name)
}

# =============================================================================
# Save convergence report and fixed effects (all model pairs, one CSV each)
# =============================================================================
save_csv(bind_rows(all_convergence), "1_pupilaccept_comparison_convergence.csv")
save_csv(bind_rows(all_fixed_effects), "1_pupilaccept_comparison_fixed_effects.csv")

message("\nDDM PPC pupil accept comparison analysis complete.")
