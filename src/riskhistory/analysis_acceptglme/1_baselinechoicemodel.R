# =============================================================================
# Choice History Under Risk: Run the Baseline Behavior Model in R
# =============================================================================

betas_model1 <- list()

for (ds in datasets) {
  print(paste("Running baseline model for dataset:", ds))
  df_sub <- behavioral_data[behavioral_data$dataset == ds, ]

  # 1. RUN THE MODEL FOR MODEL COMPARISON
  model1 <- glmer(accept ~ gain_centered + loss_centered +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)

  # 3. Add models to list
  models <- list(
    M1_part1 = model1
  )

  behavioral_models_list <- add_models_to_list(
    models_list = behavioral_models_list,
    ds     = ds,
    models = models
  )

  # 4. COMPUTE SUBJECTIVE VALUES FROM BASELINE MODEL
  betas_model1[[ds]] <- extract_random_effects(
    model = model1,
    ds = ds,
    coef_map = c("(Intercept)" = "intercept", "gain_centered" = "gain", "loss_centered" = "loss")
  )
}

# Combine all datasets' betas into a single data frame
final_betas_model1 <- bind_rows(betas_model1)

# Save to CSV
write.csv(final_betas_model1, file.path(analysis_results_path,
                                        "1_baselinebehavior_betas.csv"),
          row.names = FALSE)
