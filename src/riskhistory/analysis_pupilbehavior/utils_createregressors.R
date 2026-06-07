# =============================================================================
# Create regressors for pupil-behavior analysis
# =============================================================================

create_pupilbehavior_regressors <- function(df) {
  df %>%
    group_by(sub) %>%
    mutate(
      # Within-subject quantile
      pupil_bin = factor(
        dplyr::case_when(
          pupil < quantile(pupil, 1/2, na.rm = TRUE) ~ "Low",
          TRUE                                       ~ "High"
        ),
        levels = c("Low", "High")
      )
    ) %>%
    ungroup()
}
