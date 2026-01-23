add_models_to_list <- function(
  models_list,
  ds,
  models,
  warn_on_nonconvergence = TRUE
) {

  # Ensure dataset slot exists
  if (!ds %in% names(models_list)) {
    models_list[[ds]] <- list()
  }

  for (m_name in names(models)) {

    m <- models[[m_name]]

    # Add model to list
    models_list[[ds]][[m_name]] <- m

    # Check convergence (lme4-style models)
    if (warn_on_nonconvergence &&
          inherits(m, "merMod") &&
          !is.null(m@optinfo$conv$lme4$messages)) {

      warning(
        paste(
          "Model", m_name,
          "for dataset", ds,
          "did not converge properly."
        ),
        call. = FALSE
      )
    }
  }

  models_list
}

check_convergence_all_models <- function(
  model_list,
  datasets,
  models
) {
  # Create empty dataframe to store convergence status
  convergence_df <- data.frame(
    dataset = character(),
    model = character(),
    converged = logical(),
    stringsAsFactors = FALSE
  )

  for (ds in datasets) {
    for (m_name in models) {
      # Check if model exists for this dataset
      if (ds %in% names(model_list) && m_name %in% names(model_list[[ds]])) {
        m <- model_list[[ds]][[m_name]]

        # Check convergence (lme4-style models)
        converged <- TRUE
        if (inherits(m, "merMod")) {
          if (!is.null(m@optinfo$conv$lme4$messages)) {
            converged <- FALSE
          }
        }

        # Add to dataframe
        convergence_df <- rbind(
          convergence_df,
          data.frame(
            dataset = ds,
            model = m_name,
            converged = converged,
            stringsAsFactors = FALSE
          )
        )
      }
    }
  }

  convergence_df
}