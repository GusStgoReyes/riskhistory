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
add_subjectivevalues <- function(df, pth_betas) {
  betas_sub <- read.csv(pth_betas) %>%
    select(sub, dataset, bias, theta, lambda_)

  data <- merge(df, betas_sub, by = c("sub", "dataset"))

  data <- data %>%
    group_by(dataset, sub, run) %>%
    arrange(trial) %>%
    mutate(
      # Prospect-theory subjective value, with the negative branch rescaled
      # exactly as in the fitted model's choice rule.
      sv_raw = 0.5 * gain - 0.5 * lambda_ * loss,
      sv_inv = ifelse(sv_raw > 0, sv_raw, -abs(sv_raw) / lambda_),
      sv  = bias + sv_inv,
      ev  = 0.5 * gain_centered - 0.5 * loss_centered,
      entropy          = entropy_from_sv(theta * sv),
      abs_delta_sv = abs(sv - lag(sv)),
      prev_sv  = lag(sv),
      sv_abs = abs(sv),
      delta_sv = sv - prev_sv,
      prev_ev  = lag(ev),
      delta_ev = ev - prev_ev,
      abs_delta_ev = abs(delta_ev)
    ) %>%
    ungroup() %>%
    select(-sv_raw, -sv_inv)

  return(data)
}

# Standardize behavioral regressors within dataset/subject
standardize_behavioral_regressors <- function(df) {
  df <- df %>%
    group_by(dataset, sub) %>%
    mutate(
      sv_z          = as.numeric(scale(sv,   center = TRUE, scale = TRUE)),
      prev_sv_z     = as.numeric(scale(prev_sv,   center = TRUE, scale = TRUE)),
      delta_sv_z    = as.numeric(scale(delta_sv,  center = TRUE, scale = TRUE)),
      prev_gain_z   = as.numeric(scale(prev_gain, center = TRUE, scale = TRUE)),
      prev_loss_z   = as.numeric(scale(prev_loss, center = TRUE, scale = TRUE)),
      prev_ev_z     = as.numeric(scale(prev_ev,   center = TRUE, scale = TRUE)),
      delta_ev_z    = as.numeric(scale(delta_ev,  center = TRUE, scale = TRUE)),
      sv_abs_z      = as.numeric(scale(sv_abs,    center = TRUE, scale = TRUE)),
      rt_z          = as.numeric(scale(rt,        center = TRUE, scale = TRUE)),
      entropy_z     = as.numeric(scale(entropy,   center = TRUE, scale = TRUE)),
      gain_z        = as.numeric(scale(gain,      center = TRUE, scale = TRUE)),
      loss_z        = as.numeric(scale(loss,      center = TRUE, scale = TRUE)),
      abs_delta_ev_z = as.numeric(scale(abs_delta_ev, center = TRUE, scale = TRUE)),
      sv_partition = {
        breaks <- quantile(sv, probs = c(.2, .4, .6, .8), names = FALSE,
                            na.rm = TRUE)
        cut(sv, breaks = c(-Inf, breaks, Inf),
            labels = c("1", "2", "3", "4", "5"))
      }
    ) %>%
    ungroup()

  df <- df %>% 
    group_by(dataset, sub, sv_partition) %>%
    mutate(
      mean_sv_partition_sub = mean(sv)
    ) %>%
    ungroup()

  df <- df %>% 
    group_by(dataset, sv_partition) %>%
    mutate(
      mean_sv_partition = mean(mean_sv_partition_sub)
    ) %>%
    ungroup()

  return(df)
}

add_subjectivevalues_and_regressors <- function(df, pth_betas) {
  data <- add_subjectivevalues(df, pth_betas)
  data <- subset(data, !exclude)
  standardize_behavioral_regressors(data)
}