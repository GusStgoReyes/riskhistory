# =============================================================================
# Pupil-Behavior — Accept Models (pupil x choice history, with/without RT)
# =============================================================================

# Fixed effects and per-dataset random-effects structure for each model
PUPILACCEPT_MODELS <- list(
  pupilbehavior_accept = list(
    fixed = "accept ~ gain_centered + loss_centered + choice_prev1 * pupil",
    ranef = c(
      equalIndifference = paste("(1 + gain_centered + loss_centered +",
                                "choice_prev1 + pupil | sub)"),
      equalRange        = "(1 + gain_centered + loss_centered | sub)",
      sheng2020         = paste("(1 + gain_centered + loss_centered +",
                                "choice_prev1 + choice_prev1:pupil || sub)")
    )
  ),
  pupilbehavior_accept_rt = list(
    fixed = paste("accept ~ gain_centered + loss_centered +",
                  "choice_prev1 * pupil + choice_prev1 * rt_z"),
    ranef = c(
      equalIndifference = paste("(1 + gain_centered + loss_centered +",
                                "choice_prev1 + pupil + rt_z || sub)"),
      equalRange        = "(1 + gain_centered + loss_centered + rt_z | sub)",
      sheng2020         = paste("(1 + gain_centered + loss_centered +",
                                "choice_prev1 + rt_z + choice_prev1:rt_z || sub)")
    )
  )
)

# Model used for the choice-history effect by pupil_bin
PUPILBIN_MODEL <- "pupilbehavior_accept"

save_csv <- function(df, filename) {
  write.csv(df, file.path(analysis_results_path, filename), row.names = FALSE)
  message("Saved: ", filename)
}

# =============================================================================
# Fit models, choice-history effect by pupil_bin, and High vs. Low contrast
# =============================================================================
pupilaccept_models <- list()
pupilaccept_convergence <- list()
historyeffect_list <- list()
pupilbin_contrast_list <- list()

for (ds in datasets) {
  # Median split of pupil within subject (ties go to "Low"); must be a column
  # before fitting so avg_comparisons(by = "pupil_bin") finds it in the model
  df <- data_pupilbehavior %>%
    filter(dataset == ds) %>%
    group_by(sub) %>%
    mutate(pupil_bin = ifelse(pupil > median(pupil), "High", "Low")) %>%
    ungroup()
  pupilaccept_models[[ds]] <- list()

  for (name in names(PUPILACCEPT_MODELS)) {
    spec  <- PUPILACCEPT_MODELS[[name]]
    ranef <- spec$ranef[[ds]]

    message(sprintf("\nFitting %s for dataset: %s", name, ds))
    m <- glmer(
      as.formula(paste(spec$fixed, "+", ranef)),
      data = df, family = binomial, control = GLMER_CONTROL
    )
    diag <- check_model_diagnostics(m)
    message(sprintf("  converged: %s, singular: %s, random effects: %s",
                    diag$converged, diag$singular, ranef))

    pupilaccept_models[[ds]][[name]] <- m
    convergence_row <- data.frame(
      dataset          = ds,
      model            = name,
      converged        = diag$converged,
      singular         = diag$singular,
      random_effects   = ranef,
      stringsAsFactors = FALSE
    )
    pupilaccept_convergence <- c(pupilaccept_convergence,
                                 list(convergence_row))
  }

  # P(accept | prev. accept) - P(accept | prev. reject) within each pupil_bin
  history_effect <- avg_comparisons(
    pupilaccept_models[[ds]][[PUPILBIN_MODEL]],
    variables = list(choice_prev1 = c(-1, 1)), by = "pupil_bin", re.form = NA
  )
  history_effect_df <- as.data.frame(history_effect)

  historyeffect_list[[ds]] <- history_effect_df %>%
    rename(historyeffect = estimate, SE = std.error,
           asymp.LCL = conf.low, asymp.UCL = conf.high) %>%
    mutate(dataset = ds) %>%
    select(dataset, pupil_bin, historyeffect, SE, asymp.LCL, asymp.UCL, p.value)

  # High - Low, matched by pupil_bin label rather than row order
  pupil_bin <- history_effect_df$pupil_bin
  hyp_matrix <- matrix((pupil_bin == "High") - (pupil_bin == "Low"),
                       ncol = 1, dimnames = list(NULL, "High - Low"))
  pupilbin_contrast_list[[ds]] <- hypotheses(history_effect,
                                             hypothesis = hyp_matrix) %>%
    as.data.frame() %>%
    rename(contrast = term, SE = std.error, z.ratio = statistic) %>%
    mutate(dataset = ds) %>%
    select(dataset, contrast, estimate, SE, z.ratio, p.value)
}

# =============================================================================
# Save convergence, fixed effects, history effects, and contrasts
# =============================================================================
save_csv(bind_rows(pupilaccept_convergence),
         "3_pupilacceptmodel_convergence.csv")

save_csv(
  extract_fixed_effects(
    model_list = pupilaccept_models,
    datasets   = datasets,
    models     = names(PUPILACCEPT_MODELS),
    p_method   = p_method
  ),
  "3_pupilacceptmodel_fixed_effects.csv"
)

add_p_adj <- function(df) {
  df %>%
    group_by(dataset) %>%
    mutate(p_adj_bh = p.adjust(p.value, method = p_method)) %>%
    ungroup()
}

save_csv(add_p_adj(bind_rows(historyeffect_list)),
         "3_pupilacceptmodel_historyeffect_by_pupilbin.csv")
save_csv(add_p_adj(bind_rows(pupilbin_contrast_list)),
         "3_pupilacceptmodel_pupilbin_contrast.csv")

message("\nPupil-behavior accept model analysis complete.")
