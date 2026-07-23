# =============================================================================
# Model 1: log(rt) ~ sv_abs + condition
# =============================================================================
# Fits the maximal correlated random-effects structure implied by the fixed
# effects; falls back to a decorrelated structure (||) on non-convergence/
# singular fit (see check_model_diagnostics() in
# analysis_acceptglme/utils_modelanalytics.R).
# =============================================================================

rtlme_data$condition <- factor(
  paste0(ifelse(rtlme_data$accept == 1, "accept", "reject"), "_",
         ifelse(rtlme_data$switch == 1, "switch", "repeat")),
  levels = c("reject_repeat", "reject_switch", "accept_repeat", "accept_switch")
)

rtlme_models_list <- list()
rtlme_interaction_ranef_structure <- list()
rtlme_emmeans_list <- list()
rtlme_contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT interaction model for dataset: %s", ds))
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]

  m <- lmer(
    log(rt) ~ sv_abs + condition + (1 + sv_abs + condition | sub),
    data = df_sub, control = LMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  ranef_structure <- "(1 + sv_abs + condition | sub)"

  if (!diag$converged || diag$singular) {
    message("  Full correlated random-effects structure did not converge cleanly; falling back to decorrelated (||)")
    m <- lmer(
      log(rt) ~ sv_abs + condition + (1 + sv_abs | sub),
      data = df_sub, control = LMER_CONTROL
    )
    diag <- check_model_diagnostics(m)
    ranef_structure <- "(1 + sv_abs | sub)"
  }

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef_structure))

  rtlme_models_list[[ds]] <- list(rtlme_interaction = m)
  rtlme_interaction_ranef_structure[[ds]] <- ranef_structure

  message(sprintf("  Computing emmeans for dataset: %s", ds))
  emm <- emmeans(m, specs = pairwise ~ condition, type = "response",
                 adjust = p_method)
  rtlme_emmeans_list[[ds]] <- as.data.frame(summary(emm$emmeans, type = "response")) %>%
    mutate(dataset = ds)
  rtlme_contrasts_list[[ds]] <- as.data.frame(summary(emm$contrasts, type = "response")) %>%
    mutate(dataset = ds)
}

# =============================================================================
# Fixed effects (raw model coefficients: condition dummy-coded vs reject_repeat)
# =============================================================================
message("\nExtracting fixed effects...")
interaction_fixed_effects <- extract_fixed_effects(
  model_list = rtlme_models_list,
  datasets   = datasets,
  models     = c("rtlme_interaction"),
  p_method   = p_method
)
write.csv(interaction_fixed_effects,
          file.path(analysis_results_path, "1_rtlme_interaction_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_interaction_fixed_effects.csv")

# =============================================================================
# Estimated marginal means (response scale: geometric-mean RT in seconds)
# =============================================================================
rtlme_emmeans_df <- bind_rows(rtlme_emmeans_list)
write.csv(rtlme_emmeans_df,
          file.path(analysis_results_path, "1_rtlme_interaction_emmeans.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_interaction_emmeans.csv")

# =============================================================================
# Pairwise contrasts among the 4 condition levels (response scale: ratios)
# =============================================================================
rtlme_contrasts_df <- bind_rows(rtlme_contrasts_list)
write.csv(rtlme_contrasts_df,
          file.path(analysis_results_path, "1_rtlme_interaction_contrasts.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_interaction_contrasts.csv")

# =============================================================================
# Convergence report
# =============================================================================
interaction_convergence <- do.call(rbind, lapply(datasets, function(ds) {
  d <- check_model_diagnostics(rtlme_models_list[[ds]][["rtlme_interaction"]])
  data.frame(dataset          = ds,
             model            = "rtlme_interaction",
             converged        = d$converged,
             singular         = d$singular,
             random_effects   = rtlme_interaction_ranef_structure[[ds]],
             stringsAsFactors = FALSE)
}))

write.csv(interaction_convergence,
          file.path(analysis_results_path, "1_rtlme_interaction_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 1_rtlme_interaction_convergence_report.csv")
