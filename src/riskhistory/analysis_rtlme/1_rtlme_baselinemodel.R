# =============================================================================
# Baseline RT model: response time as a function of |subjective value| only
# =============================================================================

# Fixed effects and per-dataset random-effects structure for each model
RTLME_BASELINE_MODELS <- list(
  rtlme_baseline = list(
    fixed = "rt ~ 1 + sv_abs_z",
    ranef = c(
      equalIndifference = "(1 + sv_abs_z || sub)",
      equalRange        = "(1 + sv_abs_z | sub)",
      sheng2020         = "(1 + sv_abs_z | sub)"
    )
  ),
  rtlme_switch = list(
    fixed = "rt ~ 1 + sv_abs_z + switch",
    ranef = c(
      equalIndifference = "(1 + sv_abs_z + switch | sub)",
      equalRange        = "(1 + sv_abs_z + switch | sub)",
      sheng2020         = "(1 + sv_abs_z + switch | sub)"
    )
  )
)

fit_rtlme <- function(df, fixed, ranef) {
  glmer(
    as.formula(paste(fixed, "+", ranef)),
    data = df,
    family = Gamma(link = "identity"),
    control = GLMER_CONTROL
  )
}

# =============================================================================
# Fit models
# =============================================================================
rtlme_baseline_models <- list()
rtlme_baseline_convergence <- list()

for (ds in datasets) {
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]
  rtlme_baseline_models[[ds]] <- list()

  for (name in names(RTLME_BASELINE_MODELS)) {
    spec  <- RTLME_BASELINE_MODELS[[name]]
    ranef <- spec$ranef[[ds]]

    message(sprintf("\nFitting %s for dataset: %s", name, ds))
    m <- fit_rtlme(df_sub, spec$fixed, ranef)
    diag <- check_model_diagnostics(m)
    message(sprintf("  converged: %s, singular: %s, random effects: %s",
                    diag$converged, diag$singular, ranef))

    rtlme_baseline_models[[ds]][[name]] <- m
    convergence_row <- data.frame(
      dataset          = ds,
      model            = name,
      converged        = diag$converged,
      singular         = diag$singular,
      random_effects   = ranef,
      stringsAsFactors = FALSE
    )
    rtlme_baseline_convergence <- c(rtlme_baseline_convergence,
                                    list(convergence_row))
  }
}
rtlme_baseline_convergence <- do.call(rbind, rtlme_baseline_convergence)

# =============================================================================
# Save fixed effects and convergence reports
# =============================================================================
for (name in names(RTLME_BASELINE_MODELS)) {
  fixed_effects <- extract_fixed_effects(
    model_list = rtlme_baseline_models,
    datasets   = datasets,
    models     = name,
    p_method   = p_method
  )
  fixed_effects_file <- paste0("1_", name, "_fixed_effects.csv")
  write.csv(fixed_effects, file.path(analysis_results_path, fixed_effects_file),
            row.names = FALSE)
  message("Saved: ", fixed_effects_file)

  convergence_file <- paste0("1_", name, "_convergence_report.csv")
  is_model <- rtlme_baseline_convergence$model == name
  write.csv(rtlme_baseline_convergence[is_model, ],
            file.path(analysis_results_path, convergence_file),
            row.names = FALSE)
  message("Saved: ", convergence_file)
}
