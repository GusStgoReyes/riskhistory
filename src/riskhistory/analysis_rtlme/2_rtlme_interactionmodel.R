# =============================================================================
# Model 2: log(rt) ~ sv_abs + transition
# =============================================================================
rtlme_models_list <- list()
rtlme_interaction_ranef_structure <- list()
rtlme_avgpred_list <- list()
rtlme_contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT interaction model for dataset: %s", ds))
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]

  m <- glmer(
    rt ~ sv_abs_z + transition + (1 + sv_abs_z + transition | sub),
    data = df_sub,
    family = Gamma(link = "identity"),
    control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  ranef_structure <- "(1 + sv_abs + transition | sub)"

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef_structure))

  rtlme_models_list[[ds]] <- list(rtlme_interaction = m)
  rtlme_interaction_ranef_structure[[ds]] <- ranef_structure

  message(sprintf("  Computing avg_predictions for dataset: %s", ds))
  pred <- avg_predictions(m, by = "transition", type = "response", re.form = NA)

  rtlme_avgpred_list[[ds]] <- as.data.frame(pred) %>%
    rename(avgpred = estimate, asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, transition, avgpred, asymp.LCL, asymp.UCL, p.value)

  message(sprintf("  Computing avg_comparisons for dataset: %s", ds))
  comp <- avg_comparisons(m, variables = list(transition = "pairwise"),
                          type = "response", re.form = NA)
  comp_df <- as.data.frame(comp)

  rtlme_contrasts_list[[ds]] <- data.frame(
    dataset  = ds,
    contrast = comp_df$contrast,
    estimate = comp_df$estimate,
    asymp.LCL = comp_df$conf.low,
    asymp.UCL = comp_df$conf.high,
    p.value  = comp_df$p.value,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Fixed effects
# =============================================================================
message("\nExtracting fixed effects...")
interaction_fixed_effects <- extract_fixed_effects(
  model_list = rtlme_models_list,
  datasets   = datasets,
  models     = c("rtlme_interaction"),
  p_method   = p_method
)
write.csv(interaction_fixed_effects,
          file.path(analysis_results_path, "2_rtlme_interaction_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_interaction_fixed_effects.csv")


rtlme_avgpred_df <- bind_rows(rtlme_avgpred_list)
write.csv(rtlme_avgpred_df,
          file.path(analysis_results_path, "2_rtlme_interaction_avgpred.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_interaction_avgpred.csv")

# =============================================================================
# Switch-effect contrasts
# =============================================================================
rtlme_contrasts_df <- bind_rows(rtlme_contrasts_list)
rtlme_contrasts_df$p_adj_bh <- ave(
  rtlme_contrasts_df$p.value, rtlme_contrasts_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(rtlme_contrasts_df,
          file.path(analysis_results_path, "2_rtlme_interaction_contrasts.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_interaction_contrasts.csv")

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
          file.path(analysis_results_path, "2_rtlme_interaction_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_interaction_convergence_report.csv")
