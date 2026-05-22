# Adds columns sv, prev_sv, delta_sv, abs_sv, abs_sv_scaled to the full
# data frame and then excludes the trials marked for exclusion.
add_subjectivevalues <- function(
    df,
    pth_betas) {
  # Gather the betas computed from the baseline model
  betas_sub <- read.csv(pth_betas)

  # Merge the subjective values into data
  data <- merge(df, betas_sub,
                by = c("sub", "dataset"))

  # Compute subjective value (sv) for each trial
  data$sv <- data$beta_gain * data$gain_centered +
    data$beta_loss * data$loss_centered
  data$ev <- 0.5 * data$gain_centered - 0.5 * data$loss_centered

  # Compute the prev_sv and delta_sv for each trial
  data <- data %>%
    group_by(dataset, sub, run) %>%
    arrange(trial) %>%
    mutate(
      prev_sv  = lag(sv),
      delta_sv = sv - prev_sv,
      prev_ev  = lag(ev),
      delta_ev = ev - prev_ev
    ) %>%
    ungroup()

  # Exclude trials
  data <- subset(data, !exclude)

  return(data)
}

# Standardize the regressors
create_behavioral_regressors <- function(df) {
  df <- df %>%
    group_by(dataset, sub) %>%
    mutate(
      prev_sv_z  = as.numeric(scale(prev_sv, center = TRUE, scale = TRUE)),
      delta_sv_z = as.numeric(scale(delta_sv, center = TRUE, scale = TRUE)),
      prev_gain_z = as.numeric(scale(prev_gain, center = TRUE, scale = TRUE)),
      prev_loss_z = as.numeric(scale(prev_loss, center = TRUE, scale = TRUE)),
      prev_ev_z  = as.numeric(scale(prev_ev, center = TRUE, scale = TRUE)),
      delta_ev_z = as.numeric(scale(delta_ev, center = TRUE, scale = TRUE))
    ) %>%
    ungroup()

  return(df)
}
