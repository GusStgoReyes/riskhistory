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

  convergence_df <- data.frame(
    dataset = character(),
    model = character(),
    converged = logical(),
    singular = logical(),
    stringsAsFactors = FALSE
  )

  for (ds in datasets) {
    for (m_name in models) {

      if (ds %in% names(model_list) &&
            m_name %in% names(model_list[[ds]])) {

        m <- model_list[[ds]][[m_name]]

        converged <- TRUE
        singular <- FALSE

        if (inherits(m, "merMod")) {

          conv_messages <- m@optinfo$conv$lme4$messages
          opt_code      <- m@optinfo$conv$opt
          converged <- (is.null(conv_messages) || length(conv_messages) == 0) &&
                       (is.null(opt_code)      || opt_code == 0)

          singular <- lme4::isSingular(m, tol = 1e-4)
        }

        convergence_df <- rbind(
          convergence_df,
          data.frame(
            dataset = ds,
            model = m_name,
            converged = converged,
            singular = singular,
            stringsAsFactors = FALSE
          )
        )
      }
    }
  }

  convergence_df
}