# =============================================================================
# Random-effects structure check (one structure, one dataset)
# =============================================================================
# Fits a single mixed model and reports what is needed to decide the next step
# of the random-effects reduction: convergence, singularity, warnings raised
# during the fit, and the random effect variances (smallest first). Fitting only -- no step is taken
# automatically.
#
# Usage (from an analysis folder, after source("0_config.R")):
#   source(file.path(shared_utils_path, "utils_modelanalytics.R"))
#   source(file.path(shared_utils_path, "utils_ranefcheck.R"))
#   df  <- behavioral_data[behavioral_data$dataset == "equalRange", ]
#   out <- check_ranef(
#     fixed  = "accept ~ gain_centered + loss_centered + choice_prev1",
#     ranef  = "(1 + gain_centered + loss_centered + choice_prev1 | sub)",
#     data   = df,
#     family = binomial
#   )
#
# family = NULL fits lmer (LMER_CONTROL); otherwise glmer (GLMER_CONTROL).

check_ranef <- function(fixed,
                        ranef,
                        data,
                        family  = NULL,
                        control = NULL,
                        label   = NULL,
                        zero_tol = 1e-8) {
  formula <- as.formula(paste(fixed, "+", ranef))

  # Capture every warning raised while fitting, including ones lme4 does not
  # store in optinfo (e.g. lmerTest's negative-eigenvalue Hessian warning)
  fit_warnings <- character(0)
  collect_warning <- function(w) {
    fit_warnings <<- c(fit_warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
  model <- tryCatch(
    withCallingHandlers({
      if (is.null(family)) {
        if (is.null(control)) control <- LMER_CONTROL
        lmer(formula, data = data, control = control)
      } else {
        if (is.null(control)) control <- GLMER_CONTROL
        glmer(formula, data = data, family = family, control = control)
      }
    }, warning = collect_warning),
    error = function(e) e
  )

  # A fit that errors out (e.g. "PIRLS loop resulted in NaN value") counts as
  # not converged; report it instead of stopping
  if (inherits(model, "error")) {
    cat("\n=============================================================\n")
    if (!is.null(label)) cat("Model:     ", label, "\n")
    cat("Fixed:     ", fixed, "\n")
    cat("Random:    ", ranef, "\n")
    cat("Fit FAILED:", conditionMessage(model), "\n")
    if (length(fit_warnings) > 0) {
      cat("Warnings during fit:\n")
      for (w in fit_warnings) cat("  -", w, "\n")
    }
    cat("=============================================================\n")
    return(invisible(list(model = NULL, converged = FALSE, singular = NA,
                          error = conditionMessage(model),
                          warnings = fit_warnings)))
  }

  diag <- check_model_diagnostics(model)

  # Variances (rows without var2) and correlations (rows with var2); the
  # residual is printed but is not a candidate for removal
  vc <- as.data.frame(VarCorr(model))
  variances <- vc[is.na(vc$var2) & vc$grp != "Residual", c("grp", "var1", "vcov", "sdcor")]
  names(variances) <- c("group", "term", "variance", "sd")
  variances <- variances[order(variances$variance), ]
  rownames(variances) <- NULL
  residual <- vc[vc$grp == "Residual", "vcov"]
  correlations <- vc[!is.na(vc$var2), c("grp", "var1", "var2", "sdcor")]
  names(correlations) <- c("group", "term1", "term2", "correlation")

  zero_terms <- variances$term[variances$variance <= zero_tol]
  smallest   <- variances$term[1]

  conv_messages <- model@optinfo$conv$lme4$messages
  conv_messages <- conv_messages[!grepl("boundary (singular) fit", conv_messages, fixed = TRUE)]

  cat("\n=============================================================\n")
  if (!is.null(label)) cat("Model:     ", label, "\n")
  cat("Fixed:     ", fixed, "\n")
  cat("Random:    ", ranef, "\n")
  cat("N obs:     ", nobs(model), "| N groups:",
      paste(names(ngrps(model)), ngrps(model), sep = "=", collapse = ", "), "\n")
  cat("Converged: ", diag$converged, "\n")
  cat("Singular:  ", diag$singular, "\n")
  if (length(conv_messages) > 0) {
    cat("Convergence messages:\n")
    for (msg in conv_messages) cat("  -", msg, "\n")
  }
  if (length(fit_warnings) > 0) {
    cat("Warnings during fit:\n")
    for (w in fit_warnings) cat("  -", w, "\n")
  }
  cat("\nRandom-effect variances (smallest first):\n")
  print(variances, digits = 4, row.names = FALSE)
  if (length(residual) > 0) cat("Residual variance:", signif(residual, 4), "\n")
  if (nrow(correlations) > 0) {
    cat("\nRandom-effect correlations:\n")
    print(correlations, digits = 3, row.names = FALSE)
  }
  cat("\nZero variances:", if (length(zero_terms) > 0) paste(zero_terms, collapse = ", ") else "none", "\n")
  cat("Smallest variance:", smallest, "\n")
  cat("=============================================================\n")

  invisible(list(
    model        = model,
    converged    = diag$converged,
    singular     = diag$singular,
    messages     = conv_messages,
    warnings     = fit_warnings,
    variances    = variances,
    correlations = correlations,
    zero_terms   = zero_terms,
    smallest     = smallest
  ))
}
