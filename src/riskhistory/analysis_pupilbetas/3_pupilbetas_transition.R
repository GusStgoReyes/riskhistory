# =============================================================================
# Pupil Betas — Transition Model
# =============================================================================

formula_pupilbetas_transition <- pupil ~ logRT_z + gain_z + loss_z +
  entropy_c + abs_delta_ev_z + transition + (1 | sub)

# =============================================================================
# Fit per dataset (REML = TRUE -- this is the final reporting model)
# =============================================================================
transitionmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting transition model for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  pupilbetas_transition <- lmer(formula_pupilbetas_transition, data = df,
                                REML = TRUE, control = LMER_CONTROL)
  print(summary(pupilbetas_transition)$varcor)

  transitionmodel_list[[ds]] <- list(pupilbetas_transition = pupilbetas_transition)
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(transitionmodel_list, datasets,
                                               "pupilbetas_transition")

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "3_pupilbetas_transition_convergence.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_transition_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = transitionmodel_list,
  datasets   = datasets,
  models     = "pupilbetas_transition",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "3_pupilbetas_transition_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_transition_fixed_effects.csv")

# =============================================================================
# Estimated marginal means + all pairwise contrasts between transition levels
# =============================================================================
emmeans_rows <- list()
contrasts_rows <- list()

for (ds in datasets) {
  message(sprintf("\nComputing transition emmeans/contrasts for dataset: %s", ds))
  m <- transitionmodel_list[[ds]][["pupilbetas_transition"]]

  pred <- avg_predictions(m, by = "transition", re.form = NA)
  levels_in_order <- pred[["transition"]]

  emmeans_rows[[ds]] <- as.data.frame(pred) %>%
    rename(emmean = estimate, SE = std.error,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, transition, emmean, SE, asymp.LCL, asymp.UCL)

  # All C(4,2) = 6 pairwise contrasts between transition levels, via
  # marginaleffects::hypotheses()'s b1, b2, ... row-index syntax (row order
  # matches levels_in_order, i.e. the reference-grid order returned above).
  pairs <- combn(levels_in_order, 2, simplify = FALSE)
  hyp_strings <- vapply(pairs, function(p) {
    i <- match(p[1], levels_in_order)
    j <- match(p[2], levels_in_order)
    sprintf("b%d - b%d = 0", i, j)
  }, character(1))
  hyp_labels <- vapply(pairs, function(p) paste(p[1], "-", p[2]), character(1))

  hyp_results <- as.data.frame(hypotheses(pred, hypothesis = hyp_strings)) %>%
    rename(SE = std.error, z.ratio = statistic)

  contrasts_rows[[ds]] <- data.frame(
    dataset  = ds,
    contrast = hyp_labels,
    estimate = hyp_results$estimate,
    SE       = hyp_results$SE,
    z.ratio  = hyp_results$z.ratio,
    p.value  = hyp_results$p.value,
    stringsAsFactors = FALSE
  )
}

emmeans_df <- bind_rows(emmeans_rows)
write.csv(emmeans_df,
          file.path(analysis_results_path,
                    "3_pupilbetas_transition_emmeans.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_transition_emmeans.csv")

# p_adj_bh: one p.adjust() call per dataset, across all 6 pairwise contrasts
# together (a family separate from the fixed-effects p_adj_bh above).
contrasts_df <- bind_rows(contrasts_rows)
contrasts_df$p_adj_bh <- ave(contrasts_df$p.value, contrasts_df$dataset,
                             FUN = function(p) p.adjust(p, method = p_method))

write.csv(contrasts_df,
          file.path(analysis_results_path,
                    "3_pupilbetas_transition_contrasts.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_transition_contrasts.csv")

message("\nPupil transition analysis complete.")
