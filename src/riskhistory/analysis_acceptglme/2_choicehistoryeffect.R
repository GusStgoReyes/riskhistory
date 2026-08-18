# =============================================================================
# Choice History Effect by Expected-Value Partition
# =============================================================================

# Pairwise contrasts among the 3 partition-level history effects (ev_partition
# factor levels are already ordered low/medium/high via cut() in
# add_ev_partition(), so no re-ranking is needed here).
EV_PARTITION_HYPOTHESIS <- matrix(
  c(-1,  1,  0,
     0, -1,  1,
    -1,  0,  1),
  nrow = 3, ncol = 3,
  dimnames = list(NULL, c("Medium - Low", "High - Medium", "High - Low"))
)

historyeffect_predictions_list <- list()
historyeffect_contrasts_list   <- list()
historyeffect_pairwise_list    <- list()

for (ds in datasets) {
  print(paste("Running EV-partition choice-history model for dataset:", ds))
  df_sub <- behavioral_data[behavioral_data$dataset == ds, ]

  # equalRange gives a singular fit with a subject random slope on choice_prev1
  if (ds == "equalRange") {
    mod <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                   (1 + loss_centered + gain_centered | sub),
                 data = df_sub, family = binomial, control = GLMER_CONTROL)
  } else {
    mod <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                   (1 + loss_centered + gain_centered + choice_prev1 | sub),
                 data = df_sub, family = binomial, control = GLMER_CONTROL)
  }

  # Predicted P(accept) by EV partition x previous choice
  message(sprintf(
    "  Computing predicted probabilities by EV partition x prev choice: %s", ds
  ))
  preds <- avg_predictions(
    mod, by = c("ev_partition", "choice_prev1"), type = "response"
  )
  preds_df <- as.data.frame(preds) %>%
    mutate(dataset = ds) %>%
    rename(predicted = estimate, SE = std.error,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    select(dataset, ev_partition, choice_prev1, predicted, SE,
           asymp.LCL, asymp.UCL, p.value)
  historyeffect_predictions_list[[ds]] <- preds_df

  # Choice-history effect (previous accept minus previous reject) and its
  # significance within each EV partition
  message(sprintf(
    "  Computing choice-history effect by EV partition for dataset: %s", ds
  ))
  comp <- avg_comparisons(
    mod, variables = list(choice_prev1 = c(-1, 1)), by = "ev_partition"
  )
  comp_df <- as.data.frame(comp) %>%
    mutate(dataset = ds) %>%
    rename(historyeffect = estimate, SE = std.error, z.ratio = statistic,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    select(dataset, ev_partition, historyeffect, SE, z.ratio,
           asymp.LCL, asymp.UCL, p.value)
  historyeffect_contrasts_list[[ds]] <- comp_df

  # Pairwise comparison of the history effect across EV partitions (e.g. is
  # the history effect bigger at high EV than at low EV?)
  message(sprintf(
    "  Computing pairwise history-effect contrasts across partitions: %s",
    ds
  ))
  pairwise <- hypotheses(comp, hypothesis = EV_PARTITION_HYPOTHESIS)
  pairwise_df <- as.data.frame(pairwise) %>%
    mutate(dataset = ds) %>%
    rename(contrast = term, SE = std.error, z.ratio = statistic) %>%
    select(dataset, contrast, estimate, SE, z.ratio, p.value)
  historyeffect_pairwise_list[[ds]] <- pairwise_df
}

# =============================================================================
# Predicted probabilities by EV partition x previous choice
# =============================================================================
historyeffect_predictions_df <- bind_rows(historyeffect_predictions_list)
write.csv(historyeffect_predictions_df,
          file.path(analysis_results_path,
                    "2_choicehistoryeffect_predictions.csv"),
          row.names = FALSE)

# =============================================================================
# Choice-history effect by EV partition
# =============================================================================
historyeffect_contrasts_df <- bind_rows(historyeffect_contrasts_list)
historyeffect_contrasts_df$p_adj_bh <- ave(
  historyeffect_contrasts_df$p.value, historyeffect_contrasts_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(historyeffect_contrasts_df,
          file.path(analysis_results_path,
                    "2_choicehistoryeffect_contrasts.csv"),
          row.names = FALSE)

# =============================================================================
# Pairwise comparison of the history effect across EV partitions
# =============================================================================
historyeffect_pairwise_df <- bind_rows(historyeffect_pairwise_list)
historyeffect_pairwise_df$p_adj_bh <- ave(
  historyeffect_pairwise_df$p.value, historyeffect_pairwise_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(historyeffect_pairwise_df,
          file.path(analysis_results_path,
                    "2_choicehistoryeffect_pairwise.csv"),
          row.names = FALSE)

print("EV-partition choice-history effect analysis complete.")
