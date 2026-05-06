# =============================================================================
# Choice History Under Risk: Profile Likelihood over Eta (Decay Rate)
# =============================================================================
# Sweeps eta from 0.05 to 0.95, fits the 6-lag choice history GLMM at each
# value (same random effects as Model 14), and saves delta log-likelihood vs
# eta per dataset.
#
# Outputs:
#   results/analysis_acceptglme/3b_eta_profile.csv
#

if (!exists("behavioral_data")) source("0_config.R")

NUM_LAGS   <- 6
ETA_GRID   <- seq(0.05, 0.95, by = 0.05)
ETA_FIXED  <- 0.7          # the value currently fixed in 0_config.R
CI_THRESH  <- -1.92        # chi^2(1)/2 for 95% profile CI

results_list <- list()

for (ds in datasets) {
  cat(sprintf("\n=== Dataset: %s ===\n", ds))
  df_sub <- behavioral_data[behavioral_data$dataset == ds, ]
  ll_vals        <- rep(NA_real_, length(ETA_GRID))
  converged_vals <- rep(NA, length(ETA_GRID))
  singular_vals  <- rep(NA, length(ETA_GRID))

  for (i in seq_along(ETA_GRID)) {
    eta <- ETA_GRID[i]
    cat(sprintf("  eta = %.2f ... ", eta))

    decay  <- setNames(list(eta), ds)
    df_tmp <- compute_choice_history(df_sub,
                                     num_prev_trials = NUM_LAGS,
                                     decay_rate      = decay)

    m <- tryCatch(
      suppressWarnings(
        glmer(accept ~ gain_centered + loss_centered + choice_history6 +
                (1 + loss_centered | sub) + (0 + choice_history6 | sub),
              data    = df_tmp,
              family  = binomial,
              control = GLMER_CONTROL)
      ),
      error = function(e) {
        cat("FAILED\n")
        NULL
      }
    )

    if (!is.null(m)) {
      ll_vals[i]        <- as.numeric(logLik(m))
      conv_messages     <- m@optinfo$conv$lme4$messages
      opt_code          <- m@optinfo$conv$opt
      converged_vals[i] <- (is.null(conv_messages) || length(conv_messages) == 0) &&
                           (is.null(opt_code)       || opt_code == 0)
      singular_vals[i]  <- lme4::isSingular(m, tol = 1e-4)
      cat(sprintf("logLik = %.2f\n", ll_vals[i]))
    }
  }

  max_ll    <- max(ll_vals, na.rm = TRUE)
  delta_ll  <- ll_vals - max_ll
  eta_hat   <- ETA_GRID[which.max(ll_vals)]
  ci_lo     <- min(ETA_GRID[!is.na(delta_ll) & delta_ll >= CI_THRESH])
  ci_hi     <- max(ETA_GRID[!is.na(delta_ll) & delta_ll >= CI_THRESH])

  cat(sprintf("  => eta_hat = %.2f  |  95%% CI: [%.2f, %.2f]\n",
              eta_hat, ci_lo, ci_hi))

  results_list[[ds]] <- data.frame(
    dataset      = ds,
    eta          = ETA_GRID,
    loglik       = ll_vals,
    delta_loglik = delta_ll,
    eta_hat      = eta_hat,
    ci_lo        = ci_lo,
    ci_hi        = ci_hi,
    in_ci        = !is.na(delta_ll) & delta_ll >= CI_THRESH,
    converged    = converged_vals,
    singular     = singular_vals
  )
}

results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL

# =============================================================================
# Save CSV
# =============================================================================
dir.create(analysis_results_path, showWarnings = FALSE, recursive = TRUE)
write.csv(results_df,
          file.path(analysis_results_path, "3b_choicehistory_profile.csv"),
          row.names = FALSE)

cat("\nDone. Results saved to", analysis_results_path, "\n")
