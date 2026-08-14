# =============================================================================
# Per-subject switch effect on RT (two-step lm, no partial pooling)
# =============================================================================

fit_switch_lm <- function(df_sub_i) {
  converged <- TRUE

  fit <- tryCatch({
    lm(log(rt) ~ sv_abs + switch, data = df_sub_i)
  }, error = function(e) {
    converged <<- FALSE
    NULL
  })

  effect <- NA_real_
  if (!is.null(fit)) {
    coef_switch <- coef(fit)["switch"]
    if (is.na(coef_switch)) {
      converged <- FALSE
    } else {
      effect <- unname(coef_switch)
    }
  }

  list(effect = effect, converged = converged)
}

persubject_results <- list()

for (ds in datasets) {
  message(sprintf("Fitting per-subject switch-effect RT models for dataset: %s", ds))
  df_ds <- rtlme_data[rtlme_data$dataset == ds, ]
  subs <- sort(unique(df_ds$sub))

  ds_results <- lapply(subs, function(s) {
    fit_result <- fit_switch_lm(df_ds[df_ds$sub == s, ])
    data.frame(
      dataset          = ds,
      sub              = s,
      switch_effect    = fit_result$effect,
      converged        = fit_result$converged,
      stringsAsFactors = FALSE
    )
  })

  persubject_results[[ds]] <- do.call(rbind, ds_results)
}

persubject_switch_betas <- do.call(rbind, persubject_results)
rownames(persubject_switch_betas) <- NULL

n_total     <- nrow(persubject_switch_betas)
n_converged <- sum(persubject_switch_betas$converged)
message(sprintf("Per-subject switch-effect RT models: %d of %d converged cleanly.",
                n_converged, n_total))

write.csv(persubject_switch_betas,
          file.path(analysis_results_path, "3_rtlme_switchpersubject_betas.csv"),
          row.names = FALSE)
message("Saved: 3_rtlme_switchpersubject_betas.csv")
