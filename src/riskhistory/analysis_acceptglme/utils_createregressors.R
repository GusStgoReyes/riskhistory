# =============================================================================
# Helpers
# =============================================================================
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

entropy_from_sv <- function(sv) {
  p_accept <- sigmoid(sv)
  p_accept <- pmax(pmin(p_accept, 1 - 1e-10), 1e-10)
  -(p_accept * log2(p_accept) + (1 - p_accept) * log2(1 - p_accept))
}

# =============================================================================
# Behavioral regressors
# =============================================================================

# Adds sv, entropy, prev/delta sv and ev columns, then excludes flagged trials.
# sv includes the intercept so it can be passed to sigmoid for entropy.
add_subjectivevalues <- function(df, pth_betas) {
  betas_sub <- read.csv(pth_betas)

  data <- merge(df, betas_sub, by = c("sub", "dataset"))

  data <- data %>%
    group_by(dataset, sub, run) %>%
    arrange(trial) %>%
    mutate(
      sv  = beta_intercept + beta_gain * gain_centered +
              beta_loss * loss_centered,
      ev  = 0.5 * gain_centered - 0.5 * loss_centered,
      entropy          = entropy_from_sv(sv),
      log_abs_delta_sv = log(abs(sv - lag(sv)) + 1e-6),
      prev_sv  = lag(sv),
      delta_sv = sv - prev_sv,
      prev_ev  = lag(ev),
      delta_ev = ev - prev_ev
    ) %>%
    ungroup()

  data <- subset(data, !exclude)

  return(data)
}

# Standardize behavioral regressors within dataset/subject
create_behavioral_regressors <- function(df) {
  df <- df %>%
    group_by(dataset, sub) %>%
    mutate(
      prev_sv_z   = as.numeric(scale(prev_sv,   center = TRUE, scale = TRUE)),
      delta_sv_z  = as.numeric(scale(delta_sv,  center = TRUE, scale = TRUE)),
      prev_gain_z = as.numeric(scale(prev_gain, center = TRUE, scale = TRUE)),
      prev_loss_z = as.numeric(scale(prev_loss, center = TRUE, scale = TRUE)),
      prev_ev_z   = as.numeric(scale(prev_ev,   center = TRUE, scale = TRUE)),
      delta_ev_z  = as.numeric(scale(delta_ev,  center = TRUE, scale = TRUE))
    ) %>%
    ungroup()

  return(df)
}

# =============================================================================
# Pupil regressors
# =============================================================================

# Standardize pupil regressors within-subject
create_pupil_regressors <- function(df) {
  df <- df %>%
    mutate(
      logRT = log(rt),
    ) %>%
    group_by(sub) %>%
    mutate(
      log_abs_delta_sv_z    = scale(log_abs_delta_sv,
                                    center = TRUE, scale = TRUE)[, 1],
      logRT_z               = scale(logRT,   center = TRUE, scale = TRUE)[, 1],
      entropy_c             = scale(entropy, center = TRUE, scale = FALSE)[, 1],
      logstreak_z = scale(log(streak),
                          center = TRUE, scale = TRUE)[, 1],
      gain_z = scale(gain)[, 1],
      loss_z = scale(loss)[, 1],
      accept_switch = accept * switch
    ) %>%
    ungroup()

  return(df)
}

# Adds a pattern factor based on the last two choices
add_pattern <- function(df) {
  df %>%
    mutate(
      prev1   = choice_prev1 > 0,
      prev2   = choice_prev2 > 0,
      pattern = case_when(
        is.na(choice_prev1) | is.na(choice_prev2) ~ NA_character_,
        (accept == 1) == prev1 & prev1 == prev2   ~ "streak_continue",
        (accept == 1) != prev1 & prev1 == prev2   ~ "streak_break",
        (accept == 1) == prev1 & prev1 != prev2   ~ "confirm_switch",
        (accept == 1) != prev1 & prev1 != prev2   ~ "return_to_prior"
      ),
      pattern = factor(pattern,
                       levels = c("streak_continue", "streak_break",
                                  "confirm_switch", "return_to_prior"))
    ) %>%
    select(-prev1, -prev2)
}
