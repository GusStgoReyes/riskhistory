# =============================================================================
# Choice History Under Risk: Permutation Test for the Choice History Effect
# =============================================================================
# For each dataset, permute choice_prev1 within sub AND run (preserving each
# run's count of prev-accept / prev-reject trials, consistent with how
# choice_prev1 was originally computed within runs), refit the choice history
# model, and store the choice_prev1 coefficient. Repeated N_PERM times to
# build a null distribution, compared against the model fit on the real
# (unpermuted) choice_prev1.

if (!exists("behavioral_data_full")) source("0_config.R")

N_PERM <- 1000

permute_choice_prev1 <- function(df) {
  df %>%
    group_by(sub) %>%
    mutate(choice_prev1 = sample(choice_prev1)) %>%
    ungroup()
}

fit_choice_prev1_coef <- function(df_sub) {
  model <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                   (1 + gain_centered + loss_centered | sub),
                 data = df_sub, family = binomial, control = GLMER_CONTROL)
  tidy(model, effects = "fixed") %>%
    filter(term == "choice_prev1") %>%
    pull(estimate)
}

permutation_results <- list()

for (ds in datasets) {
  print(paste("Running choice history permutation test for dataset:", ds))
  df_sub <- behavioral_data[behavioral_data$dataset == ds, ]

  # Observed (unpermuted) choice_prev1 coefficient
  observed_estimate <- fit_choice_prev1_coef(df_sub)
  print(paste("Observed choice_prev1 estimate for dataset", ds, ":", observed_estimate))
  perm_estimates <- rep(NA_real_, N_PERM)
  for (i in seq_len(N_PERM)) {
    print(paste("Permutation", i, "of", N_PERM))
    df_perm <- permute_choice_prev1(df_sub)
    perm_estimates[i] <- tryCatch(
      suppressWarnings(fit_choice_prev1_coef(df_perm)),
      error = function(e) NA_real_
    )
    print(paste("Permutation", i, "estimate:", perm_estimates[i]))
  }

  n_failed <- sum(is.na(perm_estimates))
  if (n_failed > 0) {
    print(warning(paste0(n_failed, " of ", N_PERM,
                    " permutation model fits failed for dataset ", ds)))
  }

  n_valid <- sum(!is.na(perm_estimates))
  empirical_p <- (sum(abs(perm_estimates) >= abs(observed_estimate),
                      na.rm = TRUE) + 1) / (n_valid + 1)

  print(sprintf(
    "Dataset: %s, Observed choice_prev1 = %.4f, Empirical p = %.4f (%d/%d valid perms)",
    ds, observed_estimate, empirical_p, n_valid, N_PERM
  ))

  permutation_results[[ds]] <- data.frame(
    dataset               = ds,
    permutation           = seq_len(N_PERM),
    choice_prev1_estimate = perm_estimates,
    observed_estimate     = observed_estimate,
    empirical_p           = empirical_p
  )
}

final_permutation_df <- bind_rows(permutation_results)

write.csv(final_permutation_df,
          file.path(analysis_results_path, "5_choicehistory_permutation.csv"),
          row.names = FALSE)
