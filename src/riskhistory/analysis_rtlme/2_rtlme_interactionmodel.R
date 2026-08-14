# =============================================================================
# Model 2: log(rt) ~ sv_abs + condition
# =============================================================================

rtlme_data$condition <- factor(
  paste0(ifelse(rtlme_data$accept == 1, "accept", "reject"), "_",
         ifelse(rtlme_data$switch == 1, "switch", "repeat")),
  levels = c("reject_repeat", "reject_switch", "accept_repeat", "accept_switch")
)

CONDITION_PAIRS <- list(
  c("reject_switch", "reject_repeat"),
  c("accept_switch", "accept_repeat")
)

rtlme_models_list <- list()
rtlme_interaction_ranef_structure <- list()
rtlme_avgpred_list <- list()
rtlme_contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT interaction model for dataset: %s", ds))
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]

  m <- lmer(
    log(rt) ~ sv_abs + condition + (1 + sv_abs | sub),
    data = df_sub, control = LMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  ranef_structure <- "(1 + sv_abs | sub)"

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef_structure))

  rtlme_models_list[[ds]] <- list(rtlme_interaction = m)
  rtlme_interaction_ranef_structure[[ds]] <- ranef_structure

  message(sprintf("  Computing avg_predictions for dataset: %s", ds))
  pred <- avg_predictions(m, by = "condition", re.form = NA, transform = "exp")

  rtlme_avgpred_list[[ds]] <- as.data.frame(pred) %>%
    rename(avgpred = estimate, asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, condition, avgpred, asymp.LCL, asymp.UCL, p.value)

  message(sprintf("  Computing avg_comparisons for dataset: %s", ds))
  condition_labels <- vapply(CONDITION_PAIRS, function(p) paste(p[1], "-", p[2]), character(1))
  comp <- avg_comparisons(m, variables = list(condition = "pairwise"),
                          re.form = NA, transform = "exp")
  comp_df <- as.data.frame(comp)
  comp_df <- comp_df[comp_df$contrast %in% condition_labels, ]

  rtlme_contrasts_list[[ds]] <- data.frame(
    dataset  = ds,
    contrast = comp_df$contrast,
    estimate = comp_df$estimate,
    p.value  = comp_df$p.value,
    stringsAsFactors = FALSE
  )
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
          file.path(analysis_results_path, "2_rtlme_interaction_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_interaction_fixed_effects.csv")


rtlme_avgpred_df <- bind_rows(rtlme_avgpred_list)
write.csv(rtlme_avgpred_df,
          file.path(analysis_results_path, "2_rtlme_interaction_avgpred.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_interaction_avgpred.csv")

# =============================================================================
# Switch-effect contrasts (ratio scale, from avg_comparisons() with
# transform = "exp"): reject_switch - reject_repeat and accept_switch -
# accept_repeat. p_adj_bh is one p.adjust() call per dataset, across both
# contrasts together.
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
