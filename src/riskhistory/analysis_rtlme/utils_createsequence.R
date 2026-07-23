# =============================================================================
# 3-trial choice sequence regressor (current, t-1, t-2)
# =============================================================================
# Builds the "sequence" factor used by the Model 2 scripts in
# 2_rtlme_sequencemodels.R: AAA/AAR/ARA/ARR on accept trials, RRR/RRA/RAR/RAA
# on reject trials (A = accept, R = reject; first letter = current trial,
# second = t-1, third = t-2).
#
# Assumes choice_prev1/choice_prev2 encode +1 (accept) / -1 (reject) with no
# 0 sentinel values remaining (see the defensive filter in 0_config.R).
add_choice_sequence <- function(df) {
  df %>%
    mutate(
      current_letter = ifelse(accept == 1, "A", "R"),
      prev1_letter    = ifelse(choice_prev1 > 0, "A", "R"),
      prev2_letter    = ifelse(choice_prev2 > 0, "A", "R"),
      sequence = paste0(current_letter, prev1_letter, prev2_letter)
    ) %>%
    select(-current_letter, -prev1_letter, -prev2_letter)
}
