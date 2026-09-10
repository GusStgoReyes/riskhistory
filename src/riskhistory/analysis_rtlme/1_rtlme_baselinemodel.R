# =============================================================================
# Model 1: log(rt) ~ sv_abs (baseline)
# =============================================================================
# Baseline RT model: response time as a function of |subjective value| only
# =============================================================================

rtlme_baseline_models_list <- list()
rtlme_baseline_ranef_structure <- list()
rtlme_switch_models_list <- list()
rtlme_switch_ranef_structure <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT baseline model for dataset: %s", ds))
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]

  m <- glmer(
    rt ~ sv_abs_z + (1 + sv_abs_z | sub),
    data = df_sub,
    family = Gamma(link = "identity"),
    control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  ranef_structure <- "(1 + sv_abs | sub)"

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef_structure))

  rtlme_baseline_models_list[[ds]] <- list(rtlme_baseline = m)
  rtlme_baseline_ranef_structure[[ds]] <- ranef_structure

  message(sprintf("\nFitting RT baseline+switch model for dataset: %s", ds))

  m_switch <- glmer(
    rt ~ sv_abs_z + switch + (1 + sv_abs_z + switch | sub),
    data = df_sub,
    family = Gamma(link = "log"),
    control = GLMER_CONTROL
  )
  diag_switch <- check_model_diagnostics(m_switch)
  ranef_structure_switch <- "(1 + sv_abs + switch | sub)"

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag_switch$converged, diag_switch$singular,
                  ranef_structure_switch))

  rtlme_switch_models_list[[ds]] <- list(rtlme_switch = m_switch)
  rtlme_switch_ranef_structure[[ds]] <- ranef_structure_switch
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

# =============================================================================
# Fixed effects: baseline + switch model
# =============================================================================
message("\nExtracting fixed effects...")
switch_fixed_effects <- extract_fixed_effects(
  model_list = rtlme_switch_models_list,
  datasets   = datasets,
  models     = c("rtlme_switch"),
  p_method   = p_method
)
write.csv(switch_fixed_effects,
          file.path(analysis_results_path, "1_rtlme_switch_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_switch_fixed_effects.csv")

# =============================================================================
# Convergence report: baseline + switch model
# =============================================================================
switch_convergence <- do.call(rbind, lapply(datasets, function(ds) {
  d <- check_model_diagnostics(rtlme_switch_models_list[[ds]][["rtlme_switch"]])
  data.frame(dataset          = ds,
             model            = "rtlme_switch",
             converged        = d$converged,
             singular         = d$singular,
             random_effects   = rtlme_switch_ranef_structure[[ds]],
             stringsAsFactors = FALSE)
}))

write.csv(switch_convergence,
          file.path(analysis_results_path,
                    "1_rtlme_switch_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_switch_convergence_report.csv")
