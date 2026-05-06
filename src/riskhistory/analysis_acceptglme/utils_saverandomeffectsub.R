extract_random_effects_sv <- function(
  model,
  ds
) {
  # Extract fixed effects
  fixef_vals <- fixef(model)
  intercept_fixed <- fixef_vals["(Intercept)"]
  loss_fixed <- fixef_vals["loss_centered"]
  gain_fixed <- fixef_vals["gain_centered"]

  # Extract random effects for loss and intercept slopes
  ranef_vals <- ranef(model)$sub %>%
    rownames_to_column("sub") %>%
    rename(u_intercept = `(Intercept)`, u_loss = loss_centered)
  ranef_vals$sub <- as.integer(ranef_vals$sub)

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

extract_random_effects_history <- function(
  model,
  ds
) {
  # Extract fixed effects
  fixef_vals <- fixef(model)
  choice_prev_fixed <- fixef_vals["choice_prev1"]

  # Extract random slope for choice history
  ranef_vals <- ranef(model)$sub %>%
    rownames_to_column("sub") %>%
    rename(u_choice_prev = choice_prev1)
  ranef_vals$sub <- as.integer(ranef_vals$sub)

  ranef_vals$dataset <- ds
  ranef <- ranef_vals %>%
    mutate(
      beta_choice_prev = u_choice_prev + choice_prev_fixed,
    ) %>%
    select(sub, dataset, beta_choice_prev)

  ranef
}