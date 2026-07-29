# =============================================================================
# Per-subject pupil beta by transition
# =============================================================================
# Collapses trial-level pupil GLM betas (must already carry the 4-level
# `transition` factor from add_transition_regressor(), see
# analysis_pupiltimeseries/utils_createtransitionregressor.R) down to one row
# per (sub, dataset, transition), with `transition` recoded to the short
# AA/AR/RA/RR codes used throughout the pupil-betas figures (figure2.ipynb,
# figure4.ipynb): AA=accept_to_accept, AR=accept_to_reject,
# RA=reject_to_accept, RR=reject_to_reject.
# =============================================================================

TRANSITION_SHORT_CODES <- c(
  accept_to_accept = "AA",
  accept_to_reject = "AR",
  reject_to_accept = "RA",
  reject_to_reject = "RR"
)

build_pupil_transition_df <- function(data, pupil_col = "pupil") {
  data %>%
    filter(!is.na(transition)) %>%
    group_by(sub, dataset, transition) %>%
    summarise(pupil_beta = mean(.data[[pupil_col]], na.rm = TRUE), .groups = "drop") %>%
    mutate(transition = unname(TRANSITION_SHORT_CODES[as.character(transition)])) %>%
    select(sub, dataset, pupil_beta, transition)
}
