# =============================================================================
# Create pupil regressors from behavioral data
# =============================================================================
# Creates derived regressors for pupil analysis: sv, entropy, logRT
# Requires: behavioral data with gain_centered, loss_centered (from scale_stimuli)
#           and merged subject-specific betas (beta_intercept, beta_gain, beta_loss)
# =============================================================================

create_pupil_regressors <- function(df) {
  # Compute subjective value (sv) using subject-specific betas
  df <- df %>%
    mutate(
      sv = beta_intercept + beta_gain * gain_centered + beta_loss * loss_centered,
      entropy = entropy_from_sv(sv),
      logRT = log(rt)
    )

  # Ensure choice_prev1 and switch are numeric
  df <- df %>%
    mutate(
      choice_prev1 = as.numeric(choice_prev1),
      switch = as.numeric(switch),
      choice = as.numeric(choice)
    )

  return(df)
}
