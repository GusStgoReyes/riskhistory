# =============================================================================
# Interaction RT model: |subjective value| + transition (current x prev. choice)
# =============================================================================

# Random-effects structure per dataset
RTLME_INTERACTION_RANEF <- c(
  equalIndifference = "(1 + sv_abs_z + transition | sub)",
  equalRange        = "(1 + sv_abs_z + transition | sub)",
  sheng2020         = "(1 + sv_abs_z + transition | sub)"
)

save_csv <- function(df, filename) {
  write.csv(df, file.path(analysis_results_path, filename), row.names = FALSE)
  message("Saved: ", filename)
}

# =============================================================================
# Fit models, predictions, and contrasts
# =============================================================================
rtlme_interaction_models <- list()
rtlme_interaction_convergence <- list()
rtlme_avgpred_list <- list()
rtlme_contrasts_list <- list()

for (ds in datasets) {
  df_sub <- rtlme_data[rtlme_data$dataset == ds, ]
  ranef  <- RTLME_INTERACTION_RANEF[[ds]]

  message(sprintf("\nFitting rtlme_interaction for dataset: %s", ds))
  m <- glmer(
    as.formula(paste("rt ~ 1 + sv_abs_z + transition +", ranef)),
    data = df_sub,
    family = Gamma(link = "identity"),
    control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef))

  rtlme_interaction_models[[ds]] <- list(rtlme_interaction = m)
  rtlme_interaction_convergence[[ds]] <- data.frame(
    dataset          = ds,
    model            = "rtlme_interaction",
    converged        = diag$converged,
    singular         = diag$singular,
    random_effects   = ranef,
    stringsAsFactors = FALSE
  )

  # Population-level (re.form = NA) RT per transition and pairwise contrasts
  rtlme_avgpred_list[[ds]] <- avg_predictions(
    m, by = "transition", type = "response", re.form = NA
  ) %>%
    as.data.frame() %>%
    rename(avgpred = estimate, asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, transition, avgpred, asymp.LCL, asymp.UCL, p.value)

  rtlme_contrasts_list[[ds]] <- avg_comparisons(
    m, variables = list(transition = "pairwise"),
    type = "response", re.form = NA
  ) %>%
    as.data.frame() %>%
    rename(asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, contrast, estimate, asymp.LCL, asymp.UCL, p.value)
}

# =============================================================================
# Save fixed effects, predictions, contrasts, and convergence report
# =============================================================================
save_csv(
  extract_fixed_effects(
    model_list = rtlme_interaction_models,
    datasets   = datasets,
    models     = "rtlme_interaction",
    p_method   = p_method
  ),
  "2_rtlme_interaction_fixed_effects.csv"
)

save_csv(bind_rows(rtlme_avgpred_list), "2_rtlme_interaction_avgpred.csv")

save_csv(
  bind_rows(rtlme_contrasts_list) %>%
    group_by(dataset) %>%
    mutate(p_adj_bh = p.adjust(p.value, method = p_method)) %>%
    ungroup(),
  "2_rtlme_interaction_contrasts.csv"
)

save_csv(bind_rows(rtlme_interaction_convergence),
         "2_rtlme_interaction_convergence_report.csv")
