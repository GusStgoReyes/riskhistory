# Compute an exponentially decayed sum of previous choices.
#
# choice_history = sum_{k=1}^{num_prev_trials} decay_rate ^ k * choice_prev{k}
#
# Parameters:
#   df             : data.frame with columns 'dataset' and 'choice_prev1', ..., 'choice_prev{num_prev_trials}'
#   num_prev_trials: number of previous trials to include (default 3)
#   decay_rate     : named list mapping dataset -> rate, e.g. list(narps = 0.5, sheng2020 = 0.3)
#
# Returns:
#   data.frame with an added 'choice_history' column

compute_choice_history <- function(df, num_prev_trials = 3, decay_rate = NULL) {
  if (is.null(decay_rate)) {
    stop("decay_rate must be provided as a named list mapping dataset -> rate")
  }

  datasets_in_df <- unique(df$dataset)
  missing <- setdiff(datasets_in_df, names(decay_rate))
  if (length(missing) > 0) {
    stop(paste("decay_rate missing for datasets:", paste(missing, collapse = ", ")))
  }

  prev_cols <- paste0("choice_prev", seq_len(num_prev_trials))
  missing_cols <- setdiff(prev_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(paste("Missing columns in dataframe:", paste(missing_cols, collapse = ", ")))
  }

  df[[paste0("choice_history", num_prev_trials)]] <- 0.0
  for (ds in names(decay_rate)) {
    rate <- decay_rate[[ds]]
    max_choice_history <- rate / (1 - rate)
    mask <- df$dataset == ds
    weights <- (rate)^seq_len(num_prev_trials)
    df[[paste0("choice_history", num_prev_trials)]][mask] <- as.matrix(df[mask, prev_cols]) %*% weights / max_choice_history
  }

  df
}
