# =============================================================================
# DDM PPC Behavior: Pupil Accept Model per DDM
# =============================================================================
# The real-data pupil x choice-history accept model is fit separately to the
# behavior simulated from each DDM, so each DDM's pupil x choice-history effect
# can be compared to the one in the real data (analysis_pupilbehavior)

DDM_MODELS <- c(
  baseline  = "HDDM_Base_Lag1-stpdrift_Pupil",
  stp       = "HDDM_Base_Lag1-stpdrift_Pupil_stp",
  stpmain   = "HDDM_Base_Lag1-stpdrift_Pupil_stpmain",
  drift     = "HDDM_Base_Lag1-stpdrift_Pupil_drift",
  driftmain = "HDDM_Base_Lag1-stpdrift_Pupil_driftmain",
  stpdrift  = "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift"
)

PUPILACCEPT_FIXED <- "logit_mean_accept ~ gain + loss + real_choice_prev1 * pupil"

# Random-effects structure per DDM and dataset, selected in
# results/ranef_selection.md starting from the maximal structure
RANEF_MAX   <- "1 + gain + loss + real_choice_prev1 + pupil + real_choice_prev1:pupil"
RANEF_NOINT <- "1 + gain + loss + real_choice_prev1 + pupil"
PUPILACCEPT_RANEF <- list(
  baseline = c(
    equalIndifference = paste0("(", RANEF_NOINT, " | sub)"),
    equalRange        = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX,   " | sub)"),
    sheng2020         = paste0("(", RANEF_NOINT, " | sub)") #paste0("(", RANEF_MAX,   " || sub)")
  ),
  stp = c(
    equalIndifference = paste0("(", RANEF_NOINT, " | sub)"),
    equalRange        = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX,   " | sub)"),
    sheng2020         = paste0("(", RANEF_NOINT, " | sub)")
  ),
  stpmain = c(
    equalIndifference = paste0("(", RANEF_NOINT, " | sub)"),
    equalRange        = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX,   " | sub)"),
    sheng2020         = paste0("(", RANEF_NOINT, " | sub)")
  ),
  drift = c(
    equalIndifference = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX, " || sub)"),
    equalRange        = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX, " || sub)"),
    sheng2020         = paste0("(", RANEF_NOINT, " | sub)") #paste0("(", RANEF_MAX, " || sub)")
  ),
  driftmain = c(
    equalIndifference = paste0("(", RANEF_NOINT, " | sub)"),
    equalRange        = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX,   " | sub)"),
    sheng2020         = paste0("(", RANEF_NOINT, " | sub)")
  ),
  stpdrift = c(
    equalIndifference = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX, " | sub)"),
    equalRange        = paste0("(", RANEF_NOINT, " | sub)"), #paste0("(", RANEF_MAX, " | sub)"),
    sheng2020         = paste0("(", RANEF_NOINT, " | sub)") #paste0("(", RANEF_MAX, " | sub)")
  )
)

all_convergence <- list()
all_fixed_effects <- list()

for (ddm_name in names(DDM_MODELS)) {
  ddm_model <- DDM_MODELS[[ddm_name]]
  message(sprintf("\n=== DDM: %s (%s) ===", ddm_name, ddm_model))

  ddm_data <- load_ddm_ppc_data(ddm_model)
  models_list <- list()

  for (ds in datasets) {
    message(sprintf("\nFitting DDM PPC pupil accept model for dataset: %s", ds))
    df_sub <- ddm_data[ddm_data$dataset == ds, ]
    ranef  <- PUPILACCEPT_RANEF[[ddm_name]][[ds]]

    m <- lmer(
      as.formula(paste(PUPILACCEPT_FIXED, "+", ranef)),
      data = df_sub, control = LMER_CONTROL
    )
    diag <- check_model_diagnostics(m)
    message(sprintf("  converged: %s, singular: %s, random effects: %s",
                    diag$converged, diag$singular, ranef))

    models_list <- add_models_to_list(
      models_list = models_list,
      ds     = ds,
      models = list(pupilaccept = m)
    )
  }

  all_convergence[[ddm_name]] <- check_convergence_all_models(
    models_list, datasets, c("pupilaccept")
  ) %>%
    mutate(ddm = ddm_name, ddm_model = ddm_model)

  all_fixed_effects[[ddm_name]] <- extract_fixed_effects(
    model_list = models_list,
    datasets   = datasets,
    models     = "pupilaccept",
    p_method   = p_method
  ) %>%
    mutate(ddm = ddm_name, ddm_model = ddm_model)
}

# =============================================================================
# Save convergence report and fixed effects (all DDMs, one CSV each)
# =============================================================================
save_csv(bind_rows(all_convergence), "1_pupilaccept_comparison_convergence.csv")
save_csv(bind_rows(all_fixed_effects), "1_pupilaccept_comparison_fixed_effects.csv")

message("\nDDM PPC pupil accept analysis complete.")
