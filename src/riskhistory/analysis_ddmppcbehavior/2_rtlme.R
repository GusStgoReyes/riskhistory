# =============================================================================
# DDM PPC Behavior: RT LME
# =============================================================================

ddm_data <- ddm_data %>%
  mutate(
    abs_sv = abs(logit_mean_accept),
    accept = as.integer(real_choice == 1),
    choice_prev1 = real_choice_prev1
  ) %>%
  add_transition_regressor()

# (x1, x2) pairs to contrast, in add_transition_regressor()'s level naming.
TRANSITION_PAIRS <- list(
  c("accept_to_accept", "reject_to_accept"),   # A-A vs R-A
  c("reject_to_reject", "accept_to_reject")    # R-R vs A-R
)
# avg_comparisons() contrast labels follow factor level order, not the order
# given in TRANSITION_PAIRS, so match either direction ("A - B" or "B - A").
transition_labels <- unlist(lapply(TRANSITION_PAIRS, function(p) {
  c(paste(p[1], "-", p[2]), paste(p[2], "-", p[1]))
}))

models_list <- list()
contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting DDM PPC RT model for dataset: %s", ds))
  df_sub <- ddm_data[ddm_data$dataset == ds, ]

  rt_model <- lmer(log(mean_rt) ~ abs_sv + transition + (1 + abs_sv | sub),
                   data = df_sub, control = LMER_CONTROL)

  models_list <- add_models_to_list(
    models_list = models_list,
    ds     = ds,
    models = list(rtlme = rt_model)
  )

  # transform = "exp" converts the log-scale contrast back to the RT scale
  # (a ratio, e.g. 1.05 = 5% longer RT), matching
  # analysis_rtlme/2_rtlme_interactionmodel.R's convention.
  message(sprintf("  Computing transition avg_comparisons for dataset: %s", ds))
  comp <- avg_comparisons(rt_model, variables = list(transition = "pairwise"),
                          transform = "exp")
  comp_df <- as.data.frame(comp)
  comp_df <- comp_df[comp_df$contrast %in% transition_labels, ]

  contrasts_list[[ds]] <- data.frame(
    dataset   = ds,
    contrast  = comp_df$contrast,
    estimate  = comp_df$estimate,
    conf.low  = comp_df$conf.low,
    conf.high = comp_df$conf.high,
    p.value   = comp_df$p.value,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  models_list, datasets, c("rtlme")
)
write.csv(convergence_df,
          file.path(analysis_results_path, "2_rtlme_convergence.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = models_list,
  datasets   = datasets,
  models     = "rtlme",
  p_method   = p_method
)
write.csv(fixed_effects,
          file.path(analysis_results_path, "2_rtlme_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_fixed_effects.csv")

# =============================================================================
# Transition contrasts (A-A vs R-A, R-R vs A-R)
# =============================================================================
contrasts_df <- bind_rows(contrasts_list)
contrasts_df$p_adj_bh <- ave(
  contrasts_df$p.value, contrasts_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(contrasts_df,
          file.path(analysis_results_path, "2_rtlme_transition_contrasts.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_transition_contrasts.csv")

message("\nDDM PPC RT LME analysis complete.")
