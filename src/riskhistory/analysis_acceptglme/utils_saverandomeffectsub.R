extract_random_effects_sv <- function(
  model,
  ds
) {
  # Extract fixed effects
  fixef_vals <- fixef(model)
  intercept_fixed <- fixef_vals["(Intercept)"]
  loss_fixed <- fixef_vals["loss_centered"]
  gain_fixed <- fixef_vals["gain_centered"]

  # Extract random effects for loss and gain (slopes)
  ranef_vals <- ranef(model)$sub %>% rownames_to_column("sub")
  ranef_vals$sub <- as.integer(ranef_vals$sub)
  colnames(ranef_vals)[2] <- "u_intercept"
  colnames(ranef_vals)[3] <- "u_loss"
  # colnames(ranef_vals)[4] <- "u_gain"

  ranef_vals$dataset <- ds
  ranef <- ranef_vals %>%
    mutate(
      beta_intercept = intercept_fixed + u_intercept,
      beta_loss = loss_fixed + u_loss,
      beta_gain = gain_fixed # + u_gain
    ) %>%
    select(sub, dataset, beta_intercept, beta_gain, beta_loss)

  ranef
}
