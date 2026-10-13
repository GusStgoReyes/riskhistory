# =============================================================================
# Choice-history regressor: transition
# =============================================================================

# Current x previous choice (4-level factor), plus a 0/1 indicator per level
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
      ),
      transition_AA = as.integer(transition == "accept_to_accept"),
      transition_AR = as.integer(transition == "accept_to_reject"),
      transition_RA = as.integer(transition == "reject_to_accept"),
      transition_RR = as.integer(transition == "reject_to_reject")
    )

  n_na <- sum(is.na(df$transition))
  if (n_na > 0) {
    warning(sprintf(
      "%d row(s) have NA transition (unexpected choice_prev1/accept values)",
      n_na
    ))
  }

  df
}
