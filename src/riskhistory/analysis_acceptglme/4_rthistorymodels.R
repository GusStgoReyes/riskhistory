# =============================================================================
# Choice History Under Risk: Run the Response Time Models with History in R
# =============================================================================
# Are response times affected by choice history in risky decisions?
pth_betas <- file.path(analysis_results_path, "1_baselinebehavior_betas.csv")
data <- add_subjectivevalues(
  df = behavioral_data_full,
  pth_betas = pth_betas
)
for (ds in datasets) {
  print(paste0("Running RT model for dataset: ", ds))
  df_sub <- data[data$dataset == ds, ]

  # RT Model: Does previous choice influence RTs?
  model_rt <- lmer(log_rt ~ abs_sv_scaled + choice * switch +
                     (1 + abs_sv_scaled | sub) + (0 + switch | sub),
                   data = df_sub)
  models <- list(
    M_RT = model_rt
  )
  rt_models_list <- add_models_to_list(
    models_list = rt_models_list,
    ds     = ds,
    models = models
  )
}