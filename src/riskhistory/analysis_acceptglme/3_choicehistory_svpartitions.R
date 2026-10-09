# =============================================================================
# Choice History Effect by Subjective-Value Partition
# =============================================================================
# Add subjective values and regressors to data (from previous file)
pth_betas <- file.path(analysis_results_path,
                       "1_prospecttheory_baseline_betas.csv")
data <- add_subjectivevalues_and_regressors(behavioral_data_full, pth_betas)

historyeffect_predictions_list <- list()
historyeffect_contrasts_list   <- list()
svpartition_models_list        <- list()

for (ds in datasets) {
  print(paste("Running SV-partition choice-history model for dataset:", ds))
  df_sub <- data[data$dataset == ds, ]

  mod <- glmer(accept ~ sv_z * choice_prev1 + sv_abs_z * choice_prev1  +
                 (1 | sub),
               data = df_sub, family = binomial, control = GLMER_CONTROL)

  svpartition_models_list <- add_models_to_list(
    models_list = svpartition_models_list,
    ds     = ds,
    models = list(M_svpartition = mod)
  )

  # Choice history effect (previous accept minus previous reject) and its
  # significance within each SV partition
  message(sprintf(
    "  Computing choice-history effect by SV partition for dataset: %s", ds
  ))
  comp <- avg_comparisons(
    mod, variables = list(choice_prev1 = c(-1, 1)),
    by = "sv_partition", type = "response", re.form = NA
  )
  sv_partition_means <- df_sub %>%
    distinct(sv_partition, mean_sv_partition)

  comp_df <- as.data.frame(comp) %>%
    mutate(dataset = ds) %>%
    left_join(sv_partition_means, by = "sv_partition") %>%
    rename(historyeffect = estimate, SE = std.error, z.ratio = statistic,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    select(dataset, sv_partition, mean_sv_partition, historyeffect, SE,
           z.ratio, asymp.LCL, asymp.UCL, p.value)
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

# =============================================================================
# Fixed effects for the SV-partition models
# =============================================================================
print("Extracting fixed effects for SV-partition models...")
svpartition_fixed_effects <- extract_fixed_effects(
  model_list = svpartition_models_list,
  datasets   = datasets,
  models     = "M_svpartition",
  p_method   = p_method
)
write.csv(svpartition_fixed_effects,
          file.path(analysis_results_path,
                    "3_choicehistory_svpartitions_fixed_effects.csv"),
          row.names = FALSE)

print("SV-partition choice-history effect analysis complete.")
