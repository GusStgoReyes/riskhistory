# =============================================================================
# Pupil Betas — Choice Alteration Model
# =============================================================================

# Continuous (0/0.5/1) choice alteration variable
formula_pupilbetas_choicealteration_continuous <- pupil ~ logRT_z + gain_z +
  loss_z + entropy_c + abs_delta_ev_z + accept +
  choice_alteration_accept_num + choice_alteration_reject_num +
  (0 + logRT_z || sub)

# Categorical choice alteration variable (3 levels: Low, Medium, High)
formula_pupilbetas_choicealteration_categorical <- pupil ~ logRT_z + gain_z +
  loss_z + entropy_c + abs_delta_ev_z + accept +
  choice_alteration_accept + choice_alteration_reject +
  (0 + logRT_z || sub)

# =============================================================================
# Fit both models per dataset
# =============================================================================
choicealterationmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting choice alteration models for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  alteration_counts <- df %>%
    mutate(choice_type = ifelse(accept == 1, "Accept", "Reject")) %>%
    count(choice_type, choice_alteration)
  message(sprintf("  Choice alteration category counts for %s:", ds))
  print(alteration_counts)

  pupilbetas_choicealteration_continuous <- lmer(
    formula_pupilbetas_choicealteration_continuous,
    data = df, REML = FALSE, control = LMER_CONTROL
  )
  print(summary(pupilbetas_choicealteration_continuous)$varcor)

  pupilbetas_choicealteration_categorical <- lmer(
    formula_pupilbetas_choicealteration_categorical,
    data = df, REML = FALSE, control = LMER_CONTROL
  )
  print(summary(pupilbetas_choicealteration_categorical)$varcor)

  choicealterationmodel_list[[ds]] <- list(
    pupilbetas_choicealteration_continuous  = pupilbetas_choicealteration_continuous,
    pupilbetas_choicealteration_categorical = pupilbetas_choicealteration_categorical
  )
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  choicealterationmodel_list, datasets,
  c("pupilbetas_choicealteration_continuous", "pupilbetas_choicealteration_categorical")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "3_pupilbetas_choicealteration_convergence.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_choicealteration_convergence.csv")

# =============================================================================
# Fixed effects (continuous model only)
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = choicealterationmodel_list,
  datasets   = datasets,
  models     = "pupilbetas_choicealteration_continuous",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "3_pupilbetas_choicealteration_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_choicealteration_fixed_effects.csv")

# =============================================================================
# Average predictions for all 6 categories (choice_alteration x accept)
# =============================================================================
avgpred_rows <- list()
contrasts_rows <- list()

for (ds in datasets) {
  message(sprintf("\nComputing choice_alteration avg_predictions/avg_comparisons for dataset: %s", ds))
  m <- choicealterationmodel_list[[ds]][["pupilbetas_choicealteration_categorical"]]
  df <- data_betas_clean %>% filter(dataset == ds)

  pred <- avg_predictions(m, by = "choice_alteration_combo")

  avgpred_rows[[ds]] <- as.data.frame(pred) %>%
    rename(avgpred = estimate, SE = std.error,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, choice_alteration_combo, avgpred, SE, asymp.LCL, asymp.UCL)

  comp_accept <- avg_comparisons(
    m, variables = list(choice_alteration_accept = "pairwise"),
    newdata = df[df$accept == 1, ]
  )
  comp_accept_df <- as.data.frame(comp_accept)
  comp_accept_df$contrast <- paste0(comp_accept_df$contrast, " (Accept)")

  comp_reject <- avg_comparisons(
    m, variables = list(choice_alteration_reject = "pairwise"),
    newdata = df[df$accept == 0, ]
  )
  comp_reject_df <- as.data.frame(comp_reject)
  comp_reject_df$contrast <- paste0(comp_reject_df$contrast, " (Reject)")

  comp_df <- rbind(comp_accept_df, comp_reject_df)

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
                    "3_pupilbetas_choicealteration_avgpred.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_choicealteration_avgpred.csv")

# p_adj_bh
contrasts_df <- bind_rows(contrasts_rows)
contrasts_df$p_adj_bh <- ave(contrasts_df$p.value, contrasts_df$dataset,
                             FUN = function(p) p.adjust(p, method = p_method))

write.csv(contrasts_df,
          file.path(analysis_results_path,
                    "3_pupilbetas_choicealteration_contrasts.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_choicealteration_contrasts.csv")

message("\nPupil choice alteration analysis complete.")
