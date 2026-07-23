# =============================================================================
# Model 2: log(rt) ~ sequence + |sv|, fit separately on accept trials and
# reject trials
# =============================================================================
# `sequence` is the 3-trial choice pattern (current, t-1, t-2; see
# utils_createsequence.R). The reference level is the "just switched"
# sequence -- ARR for the accept-trials model (accept now, after two
# consecutive rejects), RAA for the reject-trials model (reject now, after
# two consecutive accepts) -- so the intercept represents that condition and
# the other three coefficients are contrasts against it.
#
# Fits the maximal (1 + sv_abs | sub) random-effects structure, with no
# fallback on non-convergence/singular fit -- convergence/singularity is
# still checked and recorded per dataset.
#
# Unlike Model 1, this model needs a real accept/reject choice at t-2, so
# trials with choice_prev1 == 0 or choice_prev2 == 0 (the "no trial there
# yet" sentinel -- see the NOTE in 0_config.R) are dropped here, scoped to
# this script only.
#
# As in 1_rtlme_interactionmodel.R, the response is written as log(rt)
# directly in the formula (rather than the precomputed `logRT` column) so
# that emmeans(..., type = "response") recognizes the log transform and
# back-transforms to geometric-mean RT (seconds), with the pairwise
# `sequence` contrasts reported as ratios (BH-adjusted, via p_method from
# 0_config.R -- same multiple-comparisons method used elsewhere in this
# analysis, rather than emmeans' own Tukey default).
# =============================================================================

n_before <- nrow(rtlme_data)
rtlme_seq_input <- rtlme_data %>% filter(choice_prev1 != 0, choice_prev2 != 0)
message(sprintf("Sequence models: dropped %d trial(s) with insufficient choice history (choice_prev1/2 == 0)",
                n_before - nrow(rtlme_seq_input)))

rtlme_seq_data <- add_choice_sequence(rtlme_seq_input)

SEQUENCE_SPECS <- list(
  accept = list(levels = c("ARR", "AAA", "AAR", "ARA"), accept_value = 1),
  reject = list(levels = c("RAA", "RRR", "RRA", "RAR"), accept_value = 0)
)

sequence_models_list <- list()
sequence_ranef_structure <- list()
sequence_emmeans_list <- list()
sequence_contrasts_list <- list()

for (condition in names(SEQUENCE_SPECS)) {
  spec <- SEQUENCE_SPECS[[condition]]
  model_name <- paste0(condition, "_sequence")

  for (ds in datasets) {
    message(sprintf("\nFitting %s-trials sequence model for dataset: %s", condition, ds))
    df_sub <- rtlme_seq_data[rtlme_seq_data$dataset == ds &
                                rtlme_seq_data$accept == spec$accept_value, ]
    df_sub$sequence <- factor(df_sub$sequence, levels = spec$levels)

    m <- lmer(
      log(rt) ~ sequence + sv_abs + (1 + sv_abs | sub),
      data = df_sub, control = LMER_CONTROL
    )
    diag <- check_model_diagnostics(m)
    ranef_structure <- "(1 + sv_abs | sub)"

    message(sprintf("  converged: %s, singular: %s, random effects: %s",
                    diag$converged, diag$singular, ranef_structure))

    if (!ds %in% names(sequence_models_list)) sequence_models_list[[ds]] <- list()
    sequence_models_list[[ds]][[model_name]] <- m
    sequence_ranef_structure[[paste(ds, condition, sep = "_")]] <- ranef_structure

    message(sprintf("  Computing emmeans for %s-trials, dataset: %s", condition, ds))
    emm <- emmeans(m, specs = pairwise ~ sequence, type = "response",
                   adjust = p_method)
    list_key <- paste(ds, condition, sep = "_")
    sequence_emmeans_list[[list_key]] <- as.data.frame(summary(emm$emmeans, type = "response")) %>%
      mutate(dataset = ds, model = model_name)
    sequence_contrasts_list[[list_key]] <- as.data.frame(summary(emm$contrasts, type = "response")) %>%
      mutate(dataset = ds, model = model_name)
  }
}

# =============================================================================
# Fixed effects
# =============================================================================
message("\nExtracting fixed effects...")
sequence_fixed_effects <- extract_fixed_effects(
  model_list = sequence_models_list,
  datasets   = datasets,
  models     = c("accept_sequence", "reject_sequence"),
  p_method   = p_method
)
write.csv(sequence_fixed_effects,
          file.path(analysis_results_path, "2_rtlme_sequence_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_sequence_fixed_effects.csv")

# =============================================================================
# Estimated marginal means (response scale: geometric-mean RT in seconds)
# =============================================================================
sequence_emmeans_df <- bind_rows(sequence_emmeans_list)
write.csv(sequence_emmeans_df,
          file.path(analysis_results_path, "2_rtlme_sequence_emmeans.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_sequence_emmeans.csv")

# =============================================================================
# Pairwise contrasts among the 4 sequence levels (response scale: ratios)
# =============================================================================
sequence_contrasts_df <- bind_rows(sequence_contrasts_list)
write.csv(sequence_contrasts_df,
          file.path(analysis_results_path, "2_rtlme_sequence_contrasts.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_sequence_contrasts.csv")

# =============================================================================
# Convergence report
# =============================================================================
sequence_convergence <- do.call(rbind, lapply(datasets, function(ds) {
  do.call(rbind, lapply(names(SEQUENCE_SPECS), function(condition) {
    model_name <- paste0(condition, "_sequence")
    d <- check_model_diagnostics(sequence_models_list[[ds]][[model_name]])
    data.frame(dataset          = ds,
               model            = model_name,
               converged        = d$converged,
               singular         = d$singular,
               random_effects   = sequence_ranef_structure[[paste(ds, condition, sep = "_")]],
               stringsAsFactors = FALSE)
  }))
}))

write.csv(sequence_convergence,
          file.path(analysis_results_path, "2_rtlme_sequence_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 2_rtlme_sequence_convergence_report.csv")
