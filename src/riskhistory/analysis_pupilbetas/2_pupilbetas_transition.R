# =============================================================================
# Pupil Betas — Transition Model
# =============================================================================

PUPILBETAS_TRANSITION_FIXED <-
  "pupil ~ 1 + rt_z + gain_z + loss_z + entropy_z + abs_delta_ev_z + transition"

# Random-effects structure per dataset
PUPILBETAS_TRANSITION_RANEF <- c(
  equalIndifference = paste("(0 + rt_z + gain_z + loss_z + entropy_z +",
                            "abs_delta_ev_z + transition_AR +",
                            "transition_RA || sub)"),
  equalRange        = paste("(0 + rt_z + loss_z + entropy_z +",
                            "abs_delta_ev_z + transition_AR || sub)"),
  sheng2020         = paste("(0 + rt_z + gain_z + loss_z + abs_delta_ev_z",
                            "+ transition_AR+transition_RA || sub)")
)

# Current choice = accept vs. reject, each an equal-weight mean of 2 transitions
ACCEPT_TRANSITIONS <- c("accept_to_accept", "reject_to_accept")

save_csv <- function(df, filename) {
  write.csv(df, file.path(analysis_results_path, filename), row.names = FALSE)
  message("Saved: ", filename)
}

# =============================================================================
# Fit models, predictions, and contrasts
# =============================================================================
transition_models <- list()
transition_convergence <- list()
avgpred_list <- list()
contrasts_list <- list()
choice_contrast_list <- list()

for (ds in datasets) {
  df    <- data_betas_clean %>% filter(dataset == ds)
  ranef <- PUPILBETAS_TRANSITION_RANEF[[ds]]

  message(sprintf("\nFitting pupilbetas_transition for dataset: %s", ds))
  m <- lmer(as.formula(paste(PUPILBETAS_TRANSITION_FIXED, "+", ranef)),
            data = df, control = LMER_CONTROL)
  print(summary(m))
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s, random effects: %s",
                  diag$converged, diag$singular, ranef))
  print(summary(m)$varcor)

  transition_models[[ds]] <- list(pupilbetas_transition = m)
  transition_convergence[[ds]] <- data.frame(
    dataset          = ds,
    model            = "pupilbetas_transition",
    converged        = diag$converged,
    singular         = diag$singular,
    random_effects   = ranef,
    stringsAsFactors = FALSE
  )

  # Pupil per transition level, all pairwise contrasts, and accept - reject
  pred <- avg_predictions(m, by = "transition", type = "response", re.form = NA)
  avgpred_list[[ds]] <- pred %>%
    as.data.frame() %>%
    rename(avgpred = estimate, SE = std.error,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, transition, avgpred, SE, asymp.LCL, asymp.UCL)

  contrasts_list[[ds]] <- avg_comparisons(
    m, variables = list(transition = "pairwise")
  ) %>%
    as.data.frame() %>%
    rename(SE = std.error, z.ratio = statistic) %>%
    mutate(dataset = ds) %>%
    select(dataset, contrast, estimate, SE, z.ratio, p.value)

  is_accept <- as.data.frame(pred)$transition %in% ACCEPT_TRANSITIONS
  choice_weights <- matrix(ifelse(is_accept, 0.5, -0.5), ncol = 1,
                           dimnames = list(NULL, "accept - reject"))
  choice_contrast_list[[ds]] <- pred %>%
    hypotheses(hypothesis = choice_weights, type = "response", re.form = NA) %>%
    as.data.frame() %>%
    rename(contrast = term, SE = std.error, z.ratio = statistic) %>%
    mutate(dataset = ds) %>%
    select(dataset, contrast, estimate, SE, z.ratio, p.value)
}

# =============================================================================
# Save convergence, fixed effects, predictions, and contrasts
# =============================================================================
save_csv(bind_rows(transition_convergence),
         "3_pupilbetas_transition_convergence.csv")

save_csv(
  extract_fixed_effects(
    model_list = transition_models,
    datasets   = datasets,
    models     = "pupilbetas_transition",
    p_method   = p_method
  ),
  "3_pupilbetas_transition_fixed_effects.csv"
)

save_csv(bind_rows(avgpred_list), "2_pupilbetas_transition_emmeans.csv")

save_csv(
  bind_rows(contrasts_list) %>%
    group_by(dataset) %>%
    mutate(p_adj_bh = p.adjust(p.value, method = p_method)) %>%
    ungroup(),
  "2_pupilbetas_transition_contrasts.csv"
)

save_csv(bind_rows(choice_contrast_list),
         "2_pupilbetas_transition_contrast_choice.csv")

message("\nPupil transition analysis complete.")
