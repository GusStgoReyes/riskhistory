# =============================================================================
# Utility: Compute per-trial mean pupil size
# =============================================================================
# Loads pupil data, removes baseline, and computes mean pupil size for
# timebins between 0.5 and 2 seconds (exclusive).
# =============================================================================

compute_pupil_per_trial <- function(pupil_data_path) {
  # Load pupil data
  pupil_data <- read.csv(pupil_data_path)

  # Remove baseline from pupil size
  pupil_data$pupil_size <- pupil_data$pupil_size - pupil_data$baseline_pupil

  # Filter to timebins > 0.5 and < 2
  pupil_filtered <- pupil_data[pupil_data$timebin > -0.5 &
                                 pupil_data$timebin < 0.5, ]

  # Compute mean pupil size per trial
  pupil_per_trial <- aggregate(
    cbind(pupil_size, exclude) ~ sub + trial + dataset,
    data = pupil_filtered,
    FUN = function(x) {
      if (is.logical(x) || is.character(x)) {
        # For exclude column: TRUE if any timebin is excluded
        return(any(x == TRUE | x == "TRUE" | x == "True"))
      } else {
        # For pupil_size: compute mean
        return(mean(x, na.rm = TRUE))
      }
    }
  )

  # Rename columns for clarity
  colnames(pupil_per_trial) <- c("sub", "trial", "dataset",
                                  "pupil_size", "pupil_exclude")

  return(pupil_per_trial)
}
