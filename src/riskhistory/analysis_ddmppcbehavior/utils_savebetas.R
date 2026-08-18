# =============================================================================
# Extract per-subject BLUPs from the baseline (gain + loss) model
# =============================================================================
extract_random_effects_baseline <- function(model, ds) {
  fixef_vals <- fixef(model)
  intercept_fixed <- fixef_vals["(Intercept)"]
  gain_fixed <- fixef_vals["gain"]
  loss_fixed <- fixef_vals["loss"]

  ranef_vals <- ranef(model)$sub %>%
    rownames_to_column("sub") %>%
    rename(u_intercept = `(Intercept)`, u_gain = gain, u_loss = loss)
  ranef_vals$sub <- as.integer(ranef_vals$sub)
  ranef_vals$dataset <- ds

  ranef_vals %>%
    mutate(
      beta_intercept = intercept_fixed + u_intercept,
      beta_gain = gain_fixed + u_gain,
      beta_loss = loss_fixed + u_loss
    ) %>%
    select(sub, dataset, beta_intercept, beta_gain, beta_loss)
}
