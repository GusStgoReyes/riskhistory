# =============================================================================
# Choice History Under Risk: Supplementary Models in R
# =============================================================================
# Run supplementary behavioral models including interaction effects between
# choice history and stimuli.

for (ds in datasets) {
  print(paste("Running supplementary models for dataset:", ds))
  df_sub <- behavioral_data[behavioral_data$dataset == ds, ]

  # SUPPLEMENTARY MODEL 1: Interaction between choice history and stimuli
  # Tests whether the effect of gain/loss on current choice depends on
  # previous choice (accept vs reject)
  model_sup1 <- glmer(accept ~ gain_centered * choice_prev1 +
                        loss_centered * choice_prev1 +
                        (1 + loss_centered + gain_centered | sub),
                      data = df_sub, family = binomial, control = GLMER_CONTROL)
  models <- list(
    M_part1_Sup1 = model_sup1
  )

  behavioral_models_list <- add_models_to_list(
    models_list = behavioral_models_list,
    ds     = ds,
    models = models
  )
}