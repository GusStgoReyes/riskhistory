# =============================================================================
# Utility: Scale gain/loss variables based on dataset-specific ranges
# =============================================================================

scale_stimuli <- function(df) {
  # Define stimulus ranges per dataset
  ranges <- list(
    equalIndifference = list(gain = seq(from = 10, to = 40, by = 2),
                             loss = seq(from = 5, to = 20, by = 1)),
    equalRange        = list(gain = seq(from = 5, to = 20, by = 1),
                             loss = seq(from = 5, to = 20, by = 1)),
    sheng2020         = list(gain = seq(from = 1, to = 10, by = 1),
                             loss = seq(from = 1, to = 10, by = 1))
  )

  df_scaled <- df %>%
    rowwise() %>%
    mutate(
      gain_centered = gain - mean(ranges[[dataset]]$gain),
      loss_centered = loss - mean(ranges[[dataset]]$loss),
      z_gain = (gain - mean(ranges[[dataset]]$gain)) /
        sd(ranges[[dataset]]$gain),
      z_loss = (loss - mean(ranges[[dataset]]$loss)) /
        sd(ranges[[dataset]]$loss)
    ) %>%
    ungroup()

  return(df_scaled)
}
