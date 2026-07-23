# =============================================================================
# Choice-history transition regressor for the pupil timeseries models
# =============================================================================
# Adds a 4-level `transition` factor (current choice x previous choice),
# combining `accept` and `choice_prev1` directly, as a categorical alternative
# to fitting accept + switch + accept:switch (same information, since both
# are binary and fully saturate the 2x2 combination -- this just reads more
# directly as "the four transition types").
# =============================================================================

add_transition_regressor <- function(df) {
  df <- df %>%
    mutate(
      transition = case_when(
        choice_prev1 == 1  & accept == 1 ~ "accept_to_accept",
        choice_prev1 == 1  & accept == 0 ~ "accept_to_reject",
        choice_prev1 == -1 & accept == 1 ~ "reject_to_accept",
        choice_prev1 == -1 & accept == 0 ~ "reject_to_reject",
        TRUE ~ NA_character_
      ),
      transition = factor(
        transition,
        levels = c("accept_to_accept", "accept_to_reject",
                   "reject_to_accept", "reject_to_reject")
      )
    )

  n_na <- sum(is.na(df$transition))
  if (n_na > 0) {
    warning(sprintf("%d row(s) have NA transition (unexpected choice_prev1/accept values)", n_na))
  }

  df
}
