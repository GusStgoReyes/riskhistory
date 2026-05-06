# =============================================================================
# Create pupil regressors from behavioral data
# =============================================================================
# Creates derived regressors for pupil analysis: sv, entropy, logRT
# Requires: behavioral data with gain_centered, loss_centered (from scale_stimuli)
#           and merged subject-specific betas (beta_intercept, beta_gain, beta_loss)
# =============================================================================
# Sigmoid function for computing entropy from subjective value
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Compute entropy from subjective value
entropy_from_sv <- function(sv) {
  p_accept <- sigmoid(sv)
  entropy <- (p_accept * log2(p_accept) + (1 - p_accept) * log2(1 - p_accept))
  return(entropy)
}

add_sv <- function(df) {
  df <- df %>%
    group_by(dataset, sub, run) %>%
    arrange(trial) %>%
    mutate(
      sv = beta_intercept + beta_gain * gain_centered +
        beta_loss * loss_centered,
      entropy = entropy_from_sv(sv),
      log_abs_delta_sv = log(abs(sv - lag(sv)) + 1e-6)
    ) %>%
    ungroup()

  return(df)
}

create_pupil_regressors <- function(df) {
  df <- df %>%
    mutate(
        logRT = log(rt),
    ) %>%
    ungroup() %>%
    group_by(sub) %>%
    mutate(
      # SV: z-score within subject
      sv_z = scale(sv)[, 1],
      log_abs_delta_sv_z = scale(log_abs_delta_sv,
                                center = TRUE, scale = TRUE)[, 1],
      logRT_z = scale(logRT, center = TRUE, scale = TRUE)[, 1],
      # Entropy: center within subject
      entropy_c = scale(entropy, center = TRUE, scale = FALSE)[, 1],
      # Streaks: center within subject
      logstreak_on_switch_z = scale(logstreak_on_switch,
                                    center = TRUE, scale = TRUE)[, 1],
      logstreak_on_repeat_z = scale(logstreak_on_repeat,
                                    center = TRUE, scale = TRUE)[, 1]
    ) %>%
    ungroup()

  return(df)
}