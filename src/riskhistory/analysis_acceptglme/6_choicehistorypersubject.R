# =============================================================================
# Per-subject choice-history effect (two-step GLM, no partial pooling)
# =============================================================================
# For each dataset and each subject, fits a standalone logistic regression
# accept ~ gain_centered + loss_centered + choice_prev1 (family = binomial),
# independently of every other subject. Unlike the choice_prev1 random slope
# in 2_choicehistorymodels.R's model2_rnd (2_choicehistory_betas.csv), these
# estimates are not shrunk toward the population mean, so they're an
# unbiased (if noisier) per-subject effect -- useful for correlating against
# an external per-subject quantity (e.g. a DDM parameter) without inheriting
# shrinkage from the mixed model.
#
# Trials with choice_prev1 == 0 (the "no trial there yet" sentinel for the
# first trial of a run -- see the NA-fill in 0_config.R) are dropped first,
# since 0 there means "no history," not "previous reject."
#
# `converged` is FALSE if glm.fit's IRLS did not converge, or if R raises the
# "fitted probabilities numerically 0 or 1 occurred" warning, which signals
# (quasi-)separation and an unreliable, often huge, coefficient.
# =============================================================================

fit_choice_prev1_glm <- function(df_sub_i) {
  converged <- TRUE

  fit <- tryCatch({
    withCallingHandlers(
      glm(accept ~ gain_centered + loss_centered + choice_prev1,
          data = df_sub_i, family = binomial),
      warning = function(w) {
        if (grepl("fitted probabilities numerically 0 or 1 occurred",
                  conditionMessage(w), fixed = TRUE) ||
              grepl("algorithm did not converge",
                    conditionMessage(w), fixed = TRUE)) {
          converged <<- FALSE
        }
        invokeRestart("muffleWarning")
      }
    )
  }, error = function(e) {
    converged <<- FALSE
    NULL
  })

  effect <- NA_real_
  if (!is.null(fit)) {
    converged <- converged && isTRUE(fit$converged)
    effect <- unname(coef(fit)["choice_prev1"])
  }

  list(effect = effect, converged = converged)
}

persubject_results <- list()

for (ds in datasets) {
  message(sprintf("Fitting per-subject choice-history GLMs for dataset: %s", ds))
  df_ds <- behavioral_data[behavioral_data$dataset == ds &
                              behavioral_data$choice_prev1 != 0, ]
  subs <- sort(unique(df_ds$sub))

  ds_results <- lapply(subs, function(s) {
    fit_result <- fit_choice_prev1_glm(df_ds[df_ds$sub == s, ])
    data.frame(
      dataset              = ds,
      sub                  = s,
      choice_prev1_effect  = fit_result$effect,
      converged            = fit_result$converged,
      stringsAsFactors     = FALSE
    )
  })

  persubject_results[[ds]] <- do.call(rbind, ds_results)
}

persubject_betas <- do.call(rbind, persubject_results)
rownames(persubject_betas) <- NULL

n_total     <- nrow(persubject_betas)
n_converged <- sum(persubject_betas$converged)
message(sprintf("Per-subject choice-history GLMs: %d of %d converged cleanly.",
                n_converged, n_total))

write.csv(persubject_betas,
          file.path(analysis_results_path, "6_choicehistorypersubject_betas.csv"),
          row.names = FALSE)
message("Saved: 6_choicehistorypersubject_betas.csv")
