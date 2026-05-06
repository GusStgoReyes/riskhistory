check_model_diagnostics <- function(model) {

  converged <- TRUE
  singular <- FALSE

  if (inherits(model, "merMod")) {

    conv_messages <- model@optinfo$conv$lme4$messages
    opt_code      <- model@optinfo$conv$opt
    converged <- (is.null(conv_messages) || length(conv_messages) == 0) &&
                 (is.null(opt_code)      || opt_code == 0)
    singular <- lme4::isSingular(model, tol = 1e-4)
  }

  list(
    converged = converged,
    singular = singular
  )
}