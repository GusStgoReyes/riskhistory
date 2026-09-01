# =============================================================================
# Choice History Effect by Subjective-Value Partition
# =============================================================================
# Add subjective values and regressors to data (from previous file)
pth_betas <- file.path(analysis_results_path, "1_baselinebehavior_betas.csv")
data <- add_subjectivevalues_and_regressors(behavioral_data_full, pth_betas)
data <- subset(data, !exclude)

historyeffect_predictions_list <- list()
historyeffect_contrasts_list   <- list()

for (ds in datasets) {
  print(paste("Running SV-partition choice-history model for dataset:", ds))
  df_sub <- data[data$dataset == ds, ]

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

  # Choice-history effect (previous accept minus previous reject) and its
  # significance within each SV partition
  message(sprintf(
    "  Computing choice-history effect by SV partition for dataset: %s", ds
  ))
  comp <- avg_comparisons(
    mod, variables = list(choice_prev1 = c(-1, 1)), by = "sv_partition"
  )
  comp_df <- as.data.frame(comp) %>%
    mutate(dataset = ds) %>%
    rename(historyeffect = estimate, SE = std.error, z.ratio = statistic,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    select(dataset, sv_partition, historyeffect, SE, z.ratio,
           asymp.LCL, asymp.UCL, p.value)
  historyeffect_contrasts_list[[ds]] <- comp_df
}

# =============================================================================
# Choice-history effect by SV partition
# =============================================================================
historyeffect_contrasts_df <- bind_rows(historyeffect_contrasts_list)
historyeffect_contrasts_df$p_adj_bh <- ave(
  historyeffect_contrasts_df$p.value, historyeffect_contrasts_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)
write.csv(historyeffect_contrasts_df,
          file.path(analysis_results_path,
                    "3_choicehistory_svpartitions_contrasts.csv"),
          row.names = FALSE)

print("SV-partition choice-history effect analysis complete.")
