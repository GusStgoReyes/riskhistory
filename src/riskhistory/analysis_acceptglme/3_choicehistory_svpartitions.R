# =============================================================================
# Choice History Effect by Subjective-Value Partition
# =============================================================================
# Add subjective values and regressors to data (from previous file)
pth_betas <- file.path(analysis_results_path,
                       "1_prospecttheory_baseline_betas.csv")
data <- add_subjectivevalues_and_regressors(behavioral_data_full, pth_betas)

# Random-effects structure per dataset
SVPARTITION_RANEF <- c(
  equalIndifference = "(1 +  sv_z + choice_prev1 + sv_abs_z*choice_prev1 || sub)",
  equalRange        = "(1 + sv_z +  sv_z:choice_prev1 + sv_abs_z || sub)",
  sheng2020         = "(1 + sv_z * choice_prev1 + sv_abs_z + choice_prev1 || sub)"
)

historyeffect_predictions_list <- list()
historyeffect_contrasts_list   <- list()
svpartition_models_list        <- list()

for (ds in datasets) {
  print(paste("Running SV-partition choice-history model for dataset:", ds))
  df_sub <- data[data$dataset == ds, ]
  ranef  <- SVPARTITION_RANEF[[ds]]

  mod <- glmer(
    as.formula(paste("accept ~ sv_z * choice_prev1 + sv_abs_z * choice_prev1 +", ranef)),
    data = df_sub, family = binomial, control = GLMER_CONTROL
  )

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

# =============================================================================
# Convergence report for the SV-partition models
# =============================================================================
svpartition_convergence <- check_convergence_all_models(
  svpartition_models_list, datasets, "M_svpartition"
)
svpartition_convergence$random_effects <- SVPARTITION_RANEF[svpartition_convergence$dataset]
print(svpartition_convergence)
write.csv(svpartition_convergence,
          file.path(analysis_results_path,
                    "3_choicehistory_svpartitions_convergence.csv"),
          row.names = FALSE)

print("SV-partition choice-history effect analysis complete.")
