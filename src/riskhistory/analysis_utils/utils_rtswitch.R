# =============================================================================
# Per-subject switch effect on RT (Gamma GLM, no partial pooling)
# =============================================================================

# Switch coefficient, or NA if the fit errors, fails to converge, or drops it
fit_switch_glm <- function(df_sub) {
  fit <- tryCatch(
    glm(rt ~ sv_abs_z + switch, data = df_sub,
        family = Gamma(link = "identity")),
    error = function(e) NULL
  )
  effect <- if (!is.null(fit) && fit$converged) {
    unname(coef(fit)["switch"])
  } else {
    NA_real_
  }
  list(effect = effect, converged = !is.na(effect))
}
