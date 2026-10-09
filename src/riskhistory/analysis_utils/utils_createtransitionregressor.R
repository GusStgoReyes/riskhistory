# =============================================================================
# Choice-history regressors: transition and choice alteration
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

# Number of the previous two choices that differ from the current one:
# 0 = Low (repeats both), 1 = Medium, 2 = High (alternates from both);
# choice_alteration_num recodes this as -1/0/1.
# Also split by current choice (accept/reject), plus a 6-level grouping label
# for avg_predictions() (not a model term).
add_choicealteration_regressor <- function(df) {
  df <- df %>%
    mutate(
      choice_now     = 2 * accept - 1,  # same -1/+1 coding as choice_prev*
      n_alternations = (choice_now != na_if(choice_prev1, 0)) +
                       (choice_now != na_if(choice_prev2, 0)),
      choice_alteration_num = n_alternations - 1,
      choice_alteration     = factor(n_alternations, levels = 0:2,
                                     labels = c("Low", "Medium", "High")),
      choice_alteration_accept = replace(choice_alteration, accept == 0, "Low"),
      choice_alteration_reject = replace(choice_alteration, accept == 1, "Low"),
      choice_alteration_accept_num =
        replace(choice_alteration_num, accept == 0, 0),
      choice_alteration_reject_num =
        replace(choice_alteration_num, accept == 1, 0),
      choice_alteration_combo = ifelse(
        is.na(choice_alteration), NA_character_,
        paste0(choice_alteration, ifelse(accept == 1, "_Accept", "_Reject"))
      )
    ) %>%
    select(-choice_now, -n_alternations)

  n_na <- sum(is.na(df$choice_alteration))
  if (n_na > 0) {
    message(sprintf(
      "  %d row(s) have NA choice_alteration (< 2 valid previous trials)",
      n_na
    ))
  }

  df
}
