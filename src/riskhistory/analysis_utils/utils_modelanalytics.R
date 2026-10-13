check_model_diagnostics <- function(model) {

  converged <- TRUE
  singular <- FALSE

  if (inherits(model, "merMod")) {

    singular <- lme4::isSingular(model, tol = 1e-4)

    # lme4 also lists "boundary (singular) fit" among its convergence messages;
    # singularity is reported above, so drop it before judging convergence
    conv_messages <- model@optinfo$conv$lme4$messages
    conv_messages <- conv_messages[!grepl("boundary (singular) fit", conv_messages, fixed = TRUE)]
    opt_code      <- model@optinfo$conv$opt
    converged <- (length(conv_messages) == 0) &&
                 (is.null(opt_code) || opt_code == 0)
  }

  list(
    converged = converged,
    singular = singular
  )
}