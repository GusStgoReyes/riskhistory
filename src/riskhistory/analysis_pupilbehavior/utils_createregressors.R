# =============================================================================
# Create regressors for pupil-behavior analysis
# =============================================================================

create_pupilbehavior_regressors <- function(df) {
  df %>%
    group_by(sub) %>%
    mutate(
      # Within-subject tertile, on residualized pupil (see 0_config.R --
      # pupil_residual has RT/gain/loss/entropy/|delta EV| and each
      # subject's own baseline already regressed out)
      pupil_tertile = dplyr::ntile(pupil_residual, 3),
      pupil_bin = factor(
        dplyr::case_when(
          pupil_tertile == 1 ~ "Low",
          pupil_tertile == 2 ~ "Medium",
          pupil_tertile == 3 ~ "High"
        ),
        levels = c("Low", "Medium", "High")
      )
    ) %>%
    select(-pupil_tertile) %>%
    ungroup()
}
