# =============================================================================
# Pupil-Behavior — Accept Model
# =============================================================================
# accept ~ gain_centered + loss_centered + choice_prev1 * pupil +
#   (1 + gain_centered + loss_centered | sub)

formula_pupilbehavior_accept <- accept ~ gain_centered + loss_centered +
  choice_prev1 * pupil +
  (1 + gain_centered + loss_centered | sub)

PUPIL_TERTILE_LABELS <- c("Low", "Medium", "High")

# Pairwise contrasts among the 3 tertile-level history effects (rows are
# always ordered Low/Medium/High, since the quantile probs below are
# increasing).
PUPIL_TERTILE_HYPOTHESIS <- matrix(
  c(-1,  1,  0,
     0, -1,  1,
    -1,  0,  1),
  nrow = 3, ncol = 3,
  dimnames = list(NULL, c("Medium - Low", "High - Medium", "High - Low"))
)

acceptmodel_list <- list()
acceptmodel_historyeffect_list <- list()
acceptmodel_historyeffect_contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting accept model for dataset: %s", ds))
  df <- data_pupilbehavior %>% filter(dataset == ds)

  m <- glmer(
    formula_pupilbehavior_accept,
    data = df, family = binomial, control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  acceptmodel_list[[ds]] <- list(pupilbehavior_accept = m)

  # Choice-history effect (predicted P(accept) when the previous choice was
  # accept minus when it was reject) at the pupil tertile boundaries
  # (33rd/50th/66th percentile of this dataset's own pupil values, the same
  # values the model was fit on) -- pupil stays continuous in the model
  # itself, this only evaluates the choice_prev1 contrast at 3 representative
  # pupil values. newdata = datagrid(..., grid_type = "counterfactual") sets
  # pupil to each tertile value for every row of the actual data (rather than
  # a single synthetic row), so the comparison is averaged over the real
  # empirical distribution of gain/loss/etc. at each pupil level.
  pupil_tertiles <- unname(quantile(df$pupil, probs = c(1/3, 1/2, 2/3), na.rm = TRUE))

  message(sprintf("  Computing choice-history effect by pupil tertile for dataset: %s", ds))
  history_effect <- avg_comparisons(
    m, variables = list(choice_prev1 = c(-1, 1)),
    newdata = datagrid(pupil = pupil_tertiles, grid_type = "counterfactual"),
    by = "pupil", re.form = NA
  )
  history_effect_df <- as.data.frame(history_effect) %>%
    mutate(dataset = ds, pupil_tertile = PUPIL_TERTILE_LABELS[rank(pupil)]) %>%
    rename(pupil_value = pupil, historyeffect = estimate, SE = std.error,
           z.ratio = statistic, asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    select(dataset, pupil_tertile, pupil_value, historyeffect, SE,
           asymp.LCL, asymp.UCL, p.value)
  acceptmodel_historyeffect_list[[ds]] <- history_effect_df

  message(sprintf("  Computing pairwise contrasts of the history effect across pupil tertiles for dataset: %s", ds))
  comp <- hypotheses(history_effect, hypothesis = PUPIL_TERTILE_HYPOTHESIS)
  comp_df <- as.data.frame(comp) %>%
    mutate(dataset = ds) %>%
    rename(contrast = term, SE = std.error, z.ratio = statistic) %>%
    select(dataset, contrast, estimate, SE, z.ratio, p.value)
  acceptmodel_historyeffect_contrasts_list[[ds]] <- comp_df
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  acceptmodel_list, datasets, c("pupilbehavior_accept")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "1_pupilbehavior_acceptmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_acceptmodel_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = acceptmodel_list,
  datasets   = datasets,
  models     = "pupilbehavior_accept",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "1_pupilbehavior_acceptmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_acceptmodel_fixed_effects.csv")

# =============================================================================
# Choice-history effect by pupil tertile + pairwise contrasts
# =============================================================================
acceptmodel_historyeffect_df <- bind_rows(acceptmodel_historyeffect_list)
acceptmodel_historyeffect_df$p_adj_bh <- ave(
  acceptmodel_historyeffect_df$p.value, acceptmodel_historyeffect_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(acceptmodel_historyeffect_df,
          file.path(analysis_results_path,
                    "1_pupilbehavior_acceptmodel_historyeffect.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_acceptmodel_historyeffect.csv")

acceptmodel_historyeffect_contrasts_df <- bind_rows(acceptmodel_historyeffect_contrasts_list)
acceptmodel_historyeffect_contrasts_df$p_adj_bh <- ave(
  acceptmodel_historyeffect_contrasts_df$p.value,
  acceptmodel_historyeffect_contrasts_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(acceptmodel_historyeffect_contrasts_df,
          file.path(analysis_results_path,
                    "1_pupilbehavior_acceptmodel_historyeffect_contrasts.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_acceptmodel_historyeffect_contrasts.csv")

message("\nPupil-behavior accept model analysis complete.")
