# =============================================================================
# Pupil Data Preprocessing Pipeline v2: Sheng2020 Conversion Utilities
# =============================================================================
# Functions to convert Sheng2020 CSV data into eyeris-compatible format.
# =============================================================================

library(dplyr)
library(tibble)
library(tidyr)

# =============================================================================
# Missing Sample Interpolation
# =============================================================================
#' Function to snap the time points to the nearest grid point
#' @param t Vector of time points
#' @param grid Vector of grid time points
#' @return Vector of snapped time points
snap_to_grid <- function(t, grid) {
  idx <- findInterval(t, grid)

  idx <- pmin(idx, length(grid) - 1)

  left  <- grid[idx]
  right <- grid[idx + 1]

  ifelse(abs(t - left) <= abs(t - right), left, right)
}

#' Fill missing samples in time series data
#'
#' When the eye tracker drops samples, this function inserts NaN values
#' at the expected time points to maintain consistent sampling.
#'
#' @param data Data frame with time series
#' @param time_col Name of time column (in ms)
#' @param sampling_rate Sampling rate in Hz
#' @param tolerance Fraction of expected dt to allow for jitter
#' @return Data frame with filled samples
fill_missing_samples_grid <- function(
  data,
  time_col = "TimeSignal",
  event_col = "StimulusName",
  sampling_rate = 125,
  tolerance = 0.5
) {
  dt <- 1000 / sampling_rate  # Assumes time is in ms

  data <- data %>% arrange(.data[[time_col]])

  if (nrow(data) <= 1) return(data)

  time_vec <- data[[time_col]]

  # Ideal time grid
  t_grid <- seq(
    from = time_vec[1],
    to   = time_vec[length(time_vec)],
    by   = dt
  )

  grid <- tibble(!!time_col := t_grid)

  # Assign observed samples to nearest grid point
  mapped_times <- snap_to_grid(time_vec, t_grid)

  data_aligned <- data %>%
    mutate(
      .grid_time  = mapped_times
    ) %>%
    mutate(!!time_col := .grid_time)


  filled <- grid %>%
    left_join(data_aligned, by = time_col) %>%
    fill(!!sym(event_col), .direction = "down")

  return(filled)
}


# =============================================================================
# Sheng2020 to Eyeris Conversion
# =============================================================================

#' Convert Sheng2020 subject/run data to eyeris format
#'
#' Takes raw Sheng2020 data for a single subject/run and creates an eyeris
#' object with properly formatted timeseries and events.
#'
#' @param sub_data Data frame containing one subject's run data
#' @param sampling_rate Sampling rate in Hz
#' @return An eyeris object ready for processing
convert_sheng2020_to_eyeris <- function(sub_data, sampling_rate = 120) {
  run_id <- as.integer(sub_data$Run[1])
  sub_id <- as.integer(sub_data$Subject[1])

  # Create timeseries in eyeris format
  timeseries <- as_tibble(sub_data) %>%
    mutate(
      block = 1,
      time_orig = TimeSignal,
      time_secs = TimeSignal / 1000,
      eye_x = GazeX,
      eye_y = GazeY,
      eye = "L",
      hz = sampling_rate,
      type = "diameter",
      pupil_raw = PupilLeft
    ) %>%
    select(block, time_orig, time_secs, eye_x, eye_y, eye, hz, type, pupil_raw)

  # Extract events from stimulus transitions
  events_df <- extract_sheng2020_events(sub_data)

  # Create info structure
  info <- data.frame(sample.rate = sampling_rate)

  # Assemble eyeris object
  data <- list(
    timeseries = list(block_1 = timeseries),
    events = list(block_1 = events_df),
    latest = list(block_1 = "pupil_raw"),
    info = info
  )
  class(data) <- "eyeris"

  return(data)
}

#' Extract events from Sheng2020 stimulus codes
#'
#' Sheng2020 encodes trial info in StimulusName:
#' - Values < 100: trial number (0-indexed)
#' - Values >= 100: stimulus code where gain = code %/% 100, loss = code %% 100
#'
#' @param sub_data Data frame with Sheng2020 data
#' @return Data frame with events in eyeris format
extract_sheng2020_events <- function(sub_data) {
  # Create a logical vector for whether it's stimulus or fixation
  is_stimulus <- sub_data$StimulusName >= 101

  # Find where transitions occur
  transitions <- diff(as.integer(is_stimulus))

  # +1 means fixation -> stimulus (trial start)
  # -1 means stimulus -> fixation (response)
  starts <- which(transitions == 1) + 1
  responses <- which(transitions == -1) + 1

  # Handle edge cases
  if (length(starts) == 0 || length(responses) == 0) {
    return(tibble(
      block = integer(),
      time = numeric(),
      text = character(),
      text_unique = character()
    ))
  }

  # Event descriptors for trial starts
  # Trial number is in the fixation period before stimulus (StimulusName < 100)
  trial_numbers <- sub_data$StimulusName[starts - 1] + 1  # Convert to 1-indexed
  stim_codes <- sub_data$StimulusName[starts]
  gains <- stim_codes %/% 100
  losses <- stim_codes %% 100
  start_descriptors <- paste0("Start_Trial", trial_numbers,
                              "_Gain", gains, "_Loss", losses)
  start_times <- sub_data$TimeSignal[starts]

  # Event descriptors for responses
  # Trial number is in the fixation period after stimulus (StimulusName < 100)
  trial_numbers_resp <- sub_data$StimulusName[responses]
  stim_codes_resp <- sub_data$StimulusName[responses - 1]
  gains_resp <- stim_codes_resp %/% 100
  losses_resp <- stim_codes_resp %% 100
  response_descriptors <- paste0("Response_Trial", trial_numbers_resp,
                                 "_Gain", gains_resp, "_Loss", losses_resp)
  response_times <- sub_data$TimeSignal[responses]

  # Create events tibble
  events_df <- tibble(
    block = 1,
    time = c(start_times, response_times),
    text = c(start_descriptors, response_descriptors),
    text_unique = c(start_descriptors, response_descriptors)
  ) %>%
    arrange(time)

  return(events_df)
}

# =============================================================================
# Data Loading and Cleaning
# =============================================================================

#' Load and clean Sheng2020 data
#'
#' @param file_path Path to eye.csv file
#' @return Cleaned data frame
load_sheng2020_data <- function(file_path) {
  all_data <- read.csv(file_path) %>%
    mutate(
      # Convert -1 values to NaN (Sheng2020's missing data code)
      PupilLeft = ifelse(PupilLeft == -1, NaN, PupilLeft),
      PupilRight = ifelse(PupilRight == -1, NaN, PupilRight),
      GazeY = ifelse(GazeY == -1, NaN, GazeY),
      GazeX = ifelse(GazeX == -1, NaN, GazeX)
    )

  return(all_data)
}

#' Split Sheng2020 data by subject and run
#'
#' @param all_data Full Sheng2020 data frame
#' @return List of data frames, one per subject-run combination
split_sheng2020_by_subject_run <- function(all_data) {
  all_data %>%
    group_by(Subject, Run) %>%
    group_split(.keep = TRUE)
}
