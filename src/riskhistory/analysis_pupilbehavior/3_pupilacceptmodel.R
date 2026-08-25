# =============================================================================
# Pupil-Behavior — Accept Model
# =============================================================================

formula_pupilbehavior_accept <- accept ~ gain_centered + loss_centered +
  choice_prev1 * pupil + choice_prev1 * logRT_z +
  (1 + gain_centered + loss_centered | sub)

acceptmodel_list <- list()
acceptmodel_historyeffect_list <- list()
acceptmodel_pupilbin_contrast_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting accept model for dataset: %s", ds))
  df <- data_pupilbehavior %>% filter(dataset == ds)

  # Binary pupil_bin (median split within sub) needs to already be a column
  # on df before fitting, so avg_comparisons() below can find it in the
  # model's own stored data via by = "pupil_bin". Ties (pupil == median) go
  # to "Low".
  df <- df %>%
    group_by(sub) %>%
    mutate(pupil_bin = ifelse(pupil > median(pupil), "High", "Low")) %>%
    ungroup()

  m <- glmer(
    formula_pupilbehavior_accept,
    data = df, family = binomial, control = GLMER_CONTROL
  )
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  acceptmodel_list[[ds]] <- list(pupilbehavior_accept = m)

  # Choice-history effect (predicted P(accept) difference between a
  # previous accept and a previous reject) within each pupil_bin.
  message(sprintf("  Computing choice-history effect by pupil_bin for dataset: %s", ds))
  history_effect <- avg_comparisons(
    m, variables = list(choice_prev1 = c(-1, 1)), by = "pupil_bin"
  )
  history_effect_df <- as.data.frame(history_effect) %>%
    mutate(dataset = ds) %>%
    rename(historyeffect = estimate, SE = std.error,
           z.ratio = statistic, asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    select(dataset, pupil_bin, historyeffect, SE, asymp.LCL, asymp.UCL, p.value)
  acceptmodel_historyeffect_list[[ds]] <- history_effect_df

  # Low-vs-High contrast of that history effect. Rows are matched by the
  # pupil_bin label (not row position), since avg_comparisons()'s own
  # ordering isn't guaranteed.
  message(sprintf("  Computing Low vs High contrast for dataset: %s", ds))
  hist_df <- as.data.frame(history_effect)
  hyp_matrix <- matrix(0, nrow = nrow(hist_df), ncol = 1,
                       dimnames = list(NULL, "High - Low"))
  hyp_matrix[hist_df$pupil_bin == "High", 1] <- 1
  hyp_matrix[hist_df$pupil_bin == "Low", 1] <- -1
  comp <- hypotheses(history_effect, hypothesis = hyp_matrix)
  comp_df <- as.data.frame(comp) %>%
    mutate(dataset = ds) %>%
    rename(contrast = term, SE = std.error, z.ratio = statistic) %>%
    select(dataset, contrast, estimate, SE, z.ratio, p.value)
  acceptmodel_pupilbin_contrast_list[[ds]] <- comp_df
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  acceptmodel_list, datasets, c("pupilbehavior_accept")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "3_pupilacceptmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 3_pupilacceptmodel_convergence.csv")

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
                    "3_pupilacceptmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilacceptmodel_fixed_effects.csv")

# =============================================================================
# Choice-history effect by pupil_bin + Low-vs-High contrast
# =============================================================================
acceptmodel_historyeffect_df <- bind_rows(acceptmodel_historyeffect_list)
acceptmodel_historyeffect_df$p_adj_bh <- ave(
  acceptmodel_historyeffect_df$p.value, acceptmodel_historyeffect_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(acceptmodel_historyeffect_df,
          file.path(analysis_results_path,
                    "3_pupilacceptmodel_historyeffect_by_pupilbin.csv"),
          row.names = FALSE)
message("Saved: 3_pupilacceptmodel_historyeffect_by_pupilbin.csv")

acceptmodel_pupilbin_contrast_df <- bind_rows(acceptmodel_pupilbin_contrast_list)
acceptmodel_pupilbin_contrast_df$p_adj_bh <- ave(
  acceptmodel_pupilbin_contrast_df$p.value, acceptmodel_pupilbin_contrast_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(acceptmodel_pupilbin_contrast_df,
          file.path(analysis_results_path,
                    "3_pupilacceptmodel_pupilbin_contrast.csv"),
          row.names = FALSE)
message("Saved: 3_pupilacceptmodel_pupilbin_contrast.csv")

message("\nPupil-behavior accept model analysis complete.")
