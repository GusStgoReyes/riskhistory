# =============================================================================
# Model 1: log(rt) ~ sv_abs (baseline)
# =============================================================================
# Baseline RT model: response time as a function of |subjective value| only,
# with no choice-history predictors. Later models in this analysis build on
# this baseline by adding choice-history terms.
#
# Fits the maximal (1 + sv_abs | sub) random-effects structure, with no
# fallback on non-convergence/singular fit -- convergence/singularity is
# still checked and recorded per dataset.
# =============================================================================

rtlme_baseline_models_list <- list()
rtlme_baseline_ranef_structure <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT baseline model for dataset: %s", ds))
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]

  m <- lmer(
    log(rt) ~ sv_abs + (1 + sv_abs | sub),
    data = df_sub, control = LMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  ranef_structure <- "(1 + sv_abs | sub)"

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef_structure))

  rtlme_baseline_models_list[[ds]] <- list(rtlme_baseline = m)
  rtlme_baseline_ranef_structure[[ds]] <- ranef_structure
}

# =============================================================================
# Fixed effects
# =============================================================================
message("\nExtracting fixed effects...")
baseline_fixed_effects <- extract_fixed_effects(
  model_list = rtlme_baseline_models_list,
  datasets   = datasets,
  models     = c("rtlme_baseline"),
  p_method   = p_method
)
write.csv(baseline_fixed_effects,
          file.path(analysis_results_path, "1_rtlme_baseline_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_baseline_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================
baseline_convergence <- do.call(rbind, lapply(datasets, function(ds) {
  d <- check_model_diagnostics(rtlme_baseline_models_list[[ds]][["rtlme_baseline"]])
  data.frame(dataset          = ds,
             model            = "rtlme_baseline",
             converged        = d$converged,
             singular         = d$singular,
             random_effects   = rtlme_baseline_ranef_structure[[ds]],
             stringsAsFactors = FALSE)
}))

write.csv(baseline_convergence,
          file.path(analysis_results_path, "1_rtlme_baseline_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_baseline_convergence_report.csv")
