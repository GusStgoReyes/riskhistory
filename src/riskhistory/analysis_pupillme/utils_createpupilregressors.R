# =============================================================================
# Create pupil regressors from behavioral data
# =============================================================================
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
      sv_z = scale(sv)[, 1],
      log_abs_delta_sv_z = scale(log_abs_delta_sv,
                                 center = TRUE, scale = TRUE)[, 1],
      logRT_z = scale(logRT, center = TRUE, scale = TRUE)[, 1],
      entropy_c = scale(entropy, center = TRUE, scale = FALSE)[, 1],
      logstreak_on_switch_z = scale(logstreak_on_switch,
                                    center = TRUE, scale = TRUE)[, 1],
      logstreak_on_repeat_z = scale(logstreak_on_repeat,
                                    center = TRUE, scale = TRUE)[, 1],
      gain_z = scale(gain)[, 1],
      loss_z = scale(loss)[, 1]
    ) %>%
    ungroup()

  return(df)
}

add_pattern <- function(df) {
  df %>%
    mutate(
      prev1   = choice_prev1 > 0,
      prev2   = choice_prev2 > 0,
      pattern = case_when(
        is.na(choice_prev1) | is.na(choice_prev2) ~ NA_character_,
        (accept == 1) == prev1 & prev1 == prev2   ~ "streak_continue", #AAA
        (accept == 1) != prev1 & prev1 == prev2   ~ "streak_break", #AAB
        (accept == 1) == prev1 & prev1 != prev2   ~ "confirm_switch", #ABB
        (accept == 1) != prev1 & prev1 != prev2   ~ "return_to_prior" #ABA
      ),
      pattern = factor(pattern,
                       levels = c("streak_continue", "streak_break",
                                  "confirm_switch", "return_to_prior"))
    ) %>%
    select(-prev1, -prev2)
}