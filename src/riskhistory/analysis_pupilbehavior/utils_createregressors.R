# =============================================================================
# Create regressors for pupil-behavior analysis
# =============================================================================

create_pupilbehavior_regressors <- function(df) {
  df %>%
    group_by(sub) %>%
    mutate(
      # Within-subject quartile split: q1 / q2 / q3 / q4
      pupil_bin = factor(
        dplyr::case_when(
          pupil < quantile(pupil, 1/4, na.rm = TRUE) ~ "q1",
          pupil < quantile(pupil, 2/4, na.rm = TRUE) ~ "q2",
          pupil < quantile(pupil, 3/4, na.rm = TRUE) ~ "q3",
          TRUE                                        ~ "q4"
        ),
        levels = c("q1", "q2", "q3", "q4")
      )
    ) %>%
    ungroup()
}
