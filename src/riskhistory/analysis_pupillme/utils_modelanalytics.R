# Check model convergence
check_model_convergence <- function(model) {
  if (inherits(model, "merMod")) {
    return(is.null(model@optinfo$conv$lme4$messages))
  }
  return(TRUE)
}
