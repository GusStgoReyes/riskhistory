# =============================================================================
# Pupil Betas — Transition Model
# =============================================================================

formula_pupilbetas_transition <- pupil ~ logRT_z + gain_z + loss_z +
  entropy_c + abs_delta_ev_z + transition + (0 + logRT_z | sub)

# =============================================================================
# Fit per dataset
# =============================================================================
transitionmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting transition model for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  pupilbetas_transition <- lmer(formula_pupilbetas_transition, data = df,
                                control = LMER_CONTROL)
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
# Average predictions + all pairwise avg_comparisons() between transition
# levels
# =============================================================================
avgpred_rows <- list()
contrasts_rows <- list()

for (ds in datasets) {
  message(sprintf("\nComputing transition avg_predictions/avg_comparisons for dataset: %s", ds))
  m <- transitionmodel_list[[ds]][["pupilbetas_transition"]]
  df <- data_betas_clean %>% filter(dataset == ds)

  pred <- avg_predictions(m, by = "transition")

  avgpred_rows[[ds]] <- as.data.frame(pred) %>%
    rename(avgpred = estimate, SE = std.error,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, transition, avgpred, SE, asymp.LCL, asymp.UCL)

  comp <- avg_comparisons(m, variables = list(transition = "pairwise"))
  comp_df <- as.data.frame(comp)

  contrasts_rows[[ds]] <- data.frame(
    dataset  = ds,
    contrast = comp_df$contrast,
    estimate = comp_df$estimate,
    SE       = comp_df$std.error,
    z.ratio  = comp_df$statistic,
    p.value  = comp_df$p.value,
    stringsAsFactors = FALSE
  )
}

avgpred_df <- bind_rows(avgpred_rows)
write.csv(avgpred_df,
          file.path(analysis_results_path,
                    "2_pupilbetas_transition_emmeans.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbetas_transition_emmeans.csv")

# p_adj_bh
contrasts_df <- bind_rows(contrasts_rows)
contrasts_df$p_adj_bh <- ave(contrasts_df$p.value, contrasts_df$dataset,
                             FUN = function(p) p.adjust(p, method = p_method))

write.csv(contrasts_df,
          file.path(analysis_results_path,
                    "2_pupilbetas_transition_contrasts.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbetas_transition_contrasts.csv")

message("\nPupil transition analysis complete.")
