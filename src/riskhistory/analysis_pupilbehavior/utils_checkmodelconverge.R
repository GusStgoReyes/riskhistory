# =============================================================================
# Utility: Check model convergence
# =============================================================================
# Checks convergence for all models present in a model_list.
# =============================================================================

check_convergence_all_models <- function(model_list, datasets) {

  convergence_df <- data.frame(
    dataset   = character(),
    model     = character(),
    converged = logical(),
    stringsAsFactors = FALSE
  )

  for (ds in datasets) {
    if (!ds %in% names(model_list)) next

    for (m_name in names(model_list[[ds]])) {
      m <- model_list[[ds]][[m_name]]

      converged <- TRUE
      if (inherits(m, "merMod")) {
        if (!is.null(m@optinfo$conv$lme4$messages)) {
          converged <- FALSE
        }
      }

      convergence_df <- rbind(
        convergence_df,
        data.frame(
          dataset   = ds,
          model     = m_name,
          converged = converged,
          stringsAsFactors = FALSE
        )
      )
    }
  }

  convergence_df
}
