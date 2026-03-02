# =============================================================================
# Utility: Load, merge, clean, and transform pupil-behavior data
# =============================================================================

process_data <- function(pupil_beta_path, behavioral_data_path) {
  # ---------------------------------------------------------------------------
  # Load and filter pupil betas
  # ---------------------------------------------------------------------------
  message("Loading pupil betas...")
  pupil_betas <- read.csv(pupil_beta_path)

  # ---------------------------------------------------------------------------
  # Load behavioral data
  # ---------------------------------------------------------------------------
  message("Loading behavioral data...")
  behavioral_data <- read.csv(behavioral_data_path)
  message(sprintf("Behavioral data: %d trials", nrow(behavioral_data)))

  # ---------------------------------------------------------------------------
  # Merge pupil and behavioral data
  # ---------------------------------------------------------------------------
  message("Merging pupil and behavioral data...")
  merged_data <- merge(behavioral_data, pupil_betas,
                       by = c("sub", "trial", "dataset", "run"),
                       all.x = FALSE)
  message(sprintf("Merged data: %d trials", nrow(merged_data)))

  # ---------------------------------------------------------------------------
  # Apply exclusion criteria
  # ---------------------------------------------------------------------------
  message("Applying exclusion criteria...")
  merged_data$combined_exclude <- (merged_data$final_exclude == "True")

  n_behavioral_exclude <- sum(merged_data$exclude == "True")
  n_combined_exclude <- sum(merged_data$combined_exclude)

  message(sprintf("Behavioral exclusions: %d", n_behavioral_exclude))
  message(sprintf("Combined exclusions: %d", n_combined_exclude))

  data_clean <- merged_data[!merged_data$combined_exclude, ]
  message(sprintf("Clean data: %d trials after exclusion", nrow(data_clean)))

  # ---------------------------------------------------------------------------
  # Rescale gain and loss
  # ---------------------------------------------------------------------------
  message("Rescaling gain and loss...")
  data_clean <- scale_stimuli(data_clean)

  # ---------------------------------------------------------------------------
  # Compute choice history
  # ---------------------------------------------------------------------------
  data_clean$choice_prev1 <- as.numeric(data_clean$choice_prev1)

  data_clean <- compute_choice_history(
    data_clean,
    num_prev_trials = 1,
    decay_rate = list(equalIndifference = 0.5, equalRange = 0.5, sheng2020 = 0.5)
  )

  data_clean$ev_centered <- 0.5 * data_clean$gain_centered - 0.5 * data_clean$loss_centered

  data_clean <- data_clean %>%
    group_by(sub) %>%
    mutate(
      pupil_rescale = (pupil - min(pupil, na.rm = TRUE)) /
        (max(pupil, na.rm = TRUE) - min(pupil, na.rm = TRUE))
    ) %>%
    ungroup()

  data_clean <- data_clean %>%
    group_by(sub, dataset) %>%
    mutate(
      pupil_bin = ntile(pupil, 3)
    ) %>%
    ungroup()

  data_clean$pupil_bin <- factor(data_clean$pupil_bin,
                                 levels = c(1, 2, 3),
                                 labels = c("low", "medium", "high"))

  return(data_clean)
}
