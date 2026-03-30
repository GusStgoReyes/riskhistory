# =============================================================================
# Create regressors for pupil-behavior analysis
# =============================================================================

create_pupilbehavior_regressors <- function(df) {
  df <- df %>%
    group_by(sub, run, dataset) %>%
    arrange(trial, .by_group = TRUE) %>%
    mutate(
      # Expected value: sum of (subject-centered) gain and loss
      expected_value = gain_centered + loss_centered
    ) %>%
    ungroup() %>%
    group_by(sub) %>%
    mutate(
      sv_z             = scale(sv)[, 1],
      expected_value_z = scale(expected_value, center = TRUE, scale = TRUE)[, 1],
      gain_centered    = gain_centered,
      loss_centered    = loss_centered,
      # log-RT z-scored within subject (needed for RT collinearity control)
      logRT_z          = scale(log(rt), center = TRUE, scale = TRUE)[, 1]
    ) %>%
    ungroup()

  # RT residualization: pupil_resid = pupil with logRT variance removed,
  # within each subject. Ensures pupil effect is not a proxy for trial duration.
  df <- df %>%
    group_by(sub) %>%
    mutate(
      pupil_resid = residuals(lm(pupil ~ logRT_z, data = cur_data()))
    ) %>%
    ungroup()

  return(df)
}
