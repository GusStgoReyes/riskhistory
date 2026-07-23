# =============================================================================
# Exponentially decayed choice history
# =============================================================================
# choice_history_t = sum_{k=1}^{min(max_lag, t-1)} (1-eta) * eta^(k-1) * choice_{t-k}
#
# i.e. a decayed sum of the past `max_lag` choices at most (fewer if fewer
# trials have happened yet in the run). Trials beyond max_lag back do not
# contribute at all, so the weights sum to (1 - eta^max_lag) rather than 1
# once a full window is available.
#
# Should be called on the full (pre-exclusion) data: a trial later excluded
# for RT reasons still really happened, so it should still contribute to the
# history available to subsequent trials in the run. Apply any `!exclude`
# filtering after this, not before.

compute_choice_history <- function(df, decay_rate, max_lag = 7) {
  if (missing(decay_rate)) {
    stop("decay_rate must be provided: a single rate, or a named list/vector mapping dataset -> rate")
  }

  per_dataset <- !is.null(names(decay_rate))
  if (per_dataset) {
    datasets_in_df <- unique(df$dataset)
    missing <- setdiff(datasets_in_df, names(decay_rate))
    if (length(missing) > 0) {
      stop(paste("decay_rate missing for datasets:", paste(missing, collapse = ", ")))
    }
  }

  df %>%
    group_by(dataset, sub, run) %>%
    arrange(trial_run, .by_group = TRUE) %>%
    mutate(
      choice_history = {
        rate <- if (per_dataset) decay_rate[[as.character(dataset[1])]] else decay_rate
        # A missing/no-response trial's own choice contributes nothing to the
        # history of later trials (consistent with how the old fixed-lag
        # choice_prevN columns were NA-filled to 0).
        choice_filled <- ifelse(is.na(choice), 0, choice)
        weights <- (1 - rate) * rate^(0:(max_lag - 1))

        history <- rep(0, length(choice_filled))
        for (k in seq_len(max_lag)) {
          lag_k <- dplyr::lag(choice_filled, k)
          # lag_k is NA only when trial t-k doesn't exist yet (i.e. fewer
          # than k trials have happened in this run) -- that trial simply
          # doesn't contribute, rather than NA-ing out the whole sum.
          history <- history + ifelse(is.na(lag_k), 0, weights[k] * lag_k)
        }
        # The first trial in the run has no prior trials at all, so its
        # history is undefined (not just a sum of zero terms).
        history[1] <- NA
        history
      }
    ) %>%
    ungroup()
}
