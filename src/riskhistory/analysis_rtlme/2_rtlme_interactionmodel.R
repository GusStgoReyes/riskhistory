# =============================================================================
# Model 2: log(rt) ~ sv_abs + transition
# =============================================================================
# transition is the current x previous choice factor added in 0_config.R
# (add_transition_regressor()): accept_to_accept, accept_to_reject,
# reject_to_accept, reject_to_reject.

# (x1, x2) pairs to contrast, in add_transition_regressor()'s level naming.
# Matches analysis_ddmppcbehavior/2_rtlme.R's convention.
TRANSITION_PAIRS <- list(
  c("accept_to_accept", "reject_to_accept"),   # A-A vs R-A
  c("reject_to_reject", "accept_to_reject")    # R-R vs A-R
)
# avg_comparisons() contrast labels follow factor level order, not the order
# given in TRANSITION_PAIRS, so match either direction ("A - B" or "B - A").
transition_labels <- unlist(lapply(TRANSITION_PAIRS, function(p) {
  c(paste(p[1], "-", p[2]), paste(p[2], "-", p[1]))
}))

rtlme_models_list <- list()
rtlme_interaction_ranef_structure <- list()
rtlme_avgpred_list <- list()
rtlme_contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting RT interaction model for dataset: %s", ds))
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]

  m <- lmer(
    log(rt) ~ sv_abs + transition + (1 + sv_abs | sub),
    data = df_sub, control = LMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  ranef_structure <- "(1 + sv_abs | sub)"

  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef_structure))

  rtlme_models_list[[ds]] <- list(rtlme_interaction = m)
  rtlme_interaction_ranef_structure[[ds]] <- ranef_structure

  message(sprintf("  Computing avg_predictions for dataset: %s", ds))
  pred <- avg_predictions(m, by = "transition", transform = "exp")

  rtlme_avgpred_list[[ds]] <- as.data.frame(pred) %>%
    rename(avgpred = estimate, asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, transition, avgpred, asymp.LCL, asymp.UCL, p.value)

  message(sprintf("  Computing avg_comparisons for dataset: %s", ds))
  comp <- avg_comparisons(m, variables = list(transition = "pairwise"),
                          transform = "exp")
  comp_df <- as.data.frame(comp)
  comp_df <- comp_df[comp_df$contrast %in% transition_labels, ]

  rtlme_contrasts_list[[ds]] <- data.frame(
    dataset  = ds,
    contrast = comp_df$contrast,
    estimate = comp_df$estimate,
    p.value  = comp_df$p.value,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Fixed effects (raw model coefficients: transition dummy-coded vs
# accept_to_accept, the first factor level)
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
# transform = "exp"): accept_to_reject - reject_to_reject and
# reject_to_accept - accept_to_accept. p_adj_bh is one p.adjust() call per
# dataset, across both contrasts together.
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
