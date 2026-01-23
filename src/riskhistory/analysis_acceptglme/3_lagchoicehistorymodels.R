# =============================================================================
# Choice History Under Risk: Run the Lag Choice History Models in R
# =============================================================================
# Run behavioral models with multiple lags of choice history to test whether
# choices further back in time influence current decisions beyond the immediate
# previous choice.

for (ds in datasets) {
  print(paste("Running lag choice history models for dataset:", ds))
  df_sub <- behavioral_data[behavioral_data$dataset == ds, ]

  # MODEL 9: Choice history with 2 lags
  model9 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                    choice_prev2 + (1 + loss_centered | sub),
                  data = df_sub, family = binomial)

  # MODEL 10: Choice history with 3 lags
  model10 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                     choice_prev2 + choice_prev3 +
                     (1 + loss_centered | sub),
                   data = df_sub, family = binomial)

  # MODEL 11: Choice history with 4 lags
  model11 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                     choice_prev2 + choice_prev3 + choice_prev4 +
                     (1 + loss_centered | sub),
                   data = df_sub, family = binomial)

  # MODEL 12: Choice history with 5 lags
  model12 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                     choice_prev2 + choice_prev3 + choice_prev4 + choice_prev5 +
                     (1 + loss_centered | sub),
                   data = df_sub, family = binomial)

  models <- list(
    M9  = model9,
    M10 = model10,
    M11 = model11,
    M12 = model12
  )

  behavioral_models_list <- add_models_to_list(
    models_list = behavioral_models_list,
    ds     = ds,
    models = models
  )
}
