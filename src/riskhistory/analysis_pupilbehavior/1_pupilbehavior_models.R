# =============================================================================
# Choice History Under Risk: Pupil-Behavior GLME Analysis
# =============================================================================

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

pupilbehavior_models_list <- list()
for (ds in datasets) {
  pupilbehavior_models_list[[ds]] <- list()
}

# =============================================================================
# Fit models per dataset
# =============================================================================

for (ds in datasets) {
  message(sprintf("\nFitting pupil-behavior models for dataset: %s", ds))
  df_sub <- data_clean %>% filter(dataset == ds)

  # ------------------------------------------------------------------
  # H1: Overcoming Choice History
  # ------------------------------------------------------------------
  if (ds == "equalRange") {
    m_h1 <- glmer(
      accept ~ gain_centered + loss_centered + choice_prev1 * pupil +
        (1 + loss_centered | sub),
      data = df_sub, family = binomial, control = ctrl
    )
  } else {
    m_h1 <- glmer(
      accept ~ gain_centered + loss_centered + choice_prev1 * pupil +
        (1 + loss_centered | sub),
      data = df_sub, family = binomial, control = ctrl
    )
  }
  # ------------------------------------------------------------------
  # H2: Evidence Gain
  # ------------------------------------------------------------------
  if (ds == "equalRange") {
    m_h2 <- glmer(
      accept ~ gain_centered * pupil + loss_centered * pupil +
        + choice_prev1 + (1 + loss_centered | sub),
      data = df_sub, family = binomial, control = ctrl
    )
  } else {
    m_h2 <- glmer(
      accept ~ gain_centered * pupil + loss_centered * pupil +
        + choice_prev1 + (1 + loss_centered | sub),
      data = df_sub, family = binomial, control = ctrl
    )
  }
  # ------------------------------------------------------------------
  # Diagnostics
  # ------------------------------------------------------------------
  models_named <- list(
    H1_overcomehist   = m_h1,
    H2_evidgain = m_h2
  )

  diag <- lapply(models_named, check_model_diagnostics)
  for (nm in names(diag)) {
    message(sprintf("  %s — converged: %s, singular: %s",
                    nm, diag[[nm]]$converged, diag[[nm]]$singular))
  }

  pupilbehavior_models_list[[ds]] <- models_named
}

# =============================================================================
# Model comparison (H5_fullModel as reference)
# =============================================================================
model_names <- c("H1_overcomehist", "H2_evidgain")

# message("\nRunning model comparison...")

# model_comparison <- compare_models(
#   model_list      = pupilbehavior_models_list,
#   datasets        = datasets,
#   models          = model_names,
#   reference_model = "H5_fullModel"
# )

# Attach convergence / singularity flags
diag_rows <- lapply(datasets, function(ds) {
  lapply(model_names, function(nm) {
    d <- check_model_diagnostics(pupilbehavior_models_list[[ds]][[nm]])
    tibble(dataset = ds, model = nm,
           converged = d$converged, singular = d$singular)
  }) %>% bind_rows()
}) %>% bind_rows()

# model_comparison <- model_comparison %>%
#   left_join(diag_rows, by = c("dataset", "model"))

# write.csv(model_comparison,
#           file.path(analysis_results_path, "1_pupilbehavior_model_comparison.csv"),
#           row.names = FALSE)
# message("Saved: 1_pupilbehavior_model_comparison.csv")

# =============================================================================
# Fixed effects
# =============================================================================

message("Extracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = pupilbehavior_models_list,
  datasets   = datasets,
  models     = model_names,
  p_method   = p_method
)

# Attach convergence / singularity flags
fixed_effects <- fixed_effects %>%
  left_join(diag_rows, by = c("dataset", "model"))

write.csv(fixed_effects,
          file.path(analysis_results_path, "1_pupilbehavior_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_fixed_effects.csv")

message("\nPupil-behavior hypothesis models complete.")