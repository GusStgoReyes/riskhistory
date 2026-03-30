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
  # H1: Prediction Error — no role for pupil
  # ------------------------------------------------------------------
  m_h1 <- glmer(
    accept ~ gain_centered + loss_centered + choice_prev1 +
      (1 + loss_centered | sub),
    data = df_sub, family = binomial, control = ctrl
  )
  # ------------------------------------------------------------------
  # H2: Policy Reset — previous-trial pupil × choice history
  # ------------------------------------------------------------------
  m_h2 <- glmer(
    accept ~ gain_centered + loss_centered + choice_prev1 + choice_prev1:pupil + pupil +
      (1 + loss_centered | sub),
    data = df_sub, family = binomial, control = ctrl
  )

  # ------------------------------------------------------------------
  # H3a: Evidence Gain — current pupil × gain and loss separately
  # ------------------------------------------------------------------
  m_h3a <- glmer(
    accept ~ gain_centered + gain_centered:pupil + loss_centered +
      loss_centered:pupil + choice_prev1 +
      (1 + loss_centered | sub),
    data = df_sub, family = binomial, control = ctrl
  )

  # ------------------------------------------------------------------
  # H3b: Evidence Gain — current pupil × expected value
  # ------------------------------------------------------------------
  m_h3b <- glmer(
    accept ~ gain_centered + loss_centered + sv_z:pupil + choice_prev1 +
      (1 + loss_centered | sub),
    data = df_sub, family = binomial, control = ctrl
  )

  # ------------------------------------------------------------------
  # Diagnostics
  # ------------------------------------------------------------------
  models_named <- list(
    H1_predError   = m_h1,
    H2_policyReset = m_h2,
    H3a_evidGain   = m_h3a,
    H3b_evidGainEV = m_h3b
  )

  diag <- lapply(models_named, check_model_diagnostics)
  for (nm in names(diag)) {
    message(sprintf("  %s — converged: %s, singular: %s",
                    nm, diag[[nm]]$converged, diag[[nm]]$singular))
  }

  pupilbehavior_models_list[[ds]] <- models_named
}

# =============================================================================
# Model comparison (H1 as reference)
# =============================================================================

message("\nRunning model comparison...")
model_names <- c("H1_predError", "H2_policyReset", "H3a_evidGain", "H3b_evidGainEV")

model_comparison <- compare_models(
  model_list      = pupilbehavior_models_list,
  datasets        = datasets,
  models          = model_names,
  reference_model = "H1_predError"
)

# Attach convergence / singularity flags
diag_rows <- lapply(datasets, function(ds) {
  lapply(model_names, function(nm) {
    d <- check_model_diagnostics(pupilbehavior_models_list[[ds]][[nm]])
    tibble(dataset = ds, model = nm,
           converged = d$converged, singular = d$singular)
  }) %>% bind_rows()
}) %>% bind_rows()

model_comparison <- model_comparison %>%
  left_join(diag_rows, by = c("dataset", "model"))

write.csv(model_comparison,
          file.path(analysis_results_path, "1_pupilbehavior_model_comparison.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbehavior_model_comparison.csv")

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

# =============================================================================
# Robustness tests for the pupil → repetition effect (CBF follow-up)
# Outcome: repeat (binary: 1 = same choice as previous trial)
#
# Test 2 — RT residualization: does the pupil effect survive after removing
#           all RT-correlated variance from pupil?
#   repeat ~ pupil_resid + logRT_z + gain_centered + loss_centered + choice_prev1
#            + (1 + pupil_resid || sub)
#
# Test 3 — RT × pupil interaction: is the pupil effect stronger on slow trials?
#   repeat ~ pupil * logRT_z + gain_centered + loss_centered + choice_prev1
#            + (1 + pupil || sub)
#
# Test 4 — Quadratic pupil: is the effect non-linear (motivated by
#           equalIndifference non-monotonicity)?
#   repeat ~ pupil + I(pupil^2) + logRT_z + gain_centered + loss_centered + choice_prev1
#            + (1 + pupil || sub)
# =============================================================================

robustness_models_list <- list()
for (ds in datasets) {
  robustness_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting robustness models for dataset: %s", ds))
  df_sub <- data_clean %>% filter(dataset == ds)

  # ------------------------------------------------------------------
  # Test 2a: Baseline repetition model with raw pupil + logRT_z
  # ------------------------------------------------------------------
  m_rep_base <- glmer(
    repeat. ~ pupil + logRT_z + gain_centered + loss_centered + choice_prev1 +
      (1 + pupil || sub),
    data = df_sub, family = binomial, control = ctrl
  )

  # ------------------------------------------------------------------
  # Test 2b: RT-residualized pupil — pupil effect above and beyond RT
  # ------------------------------------------------------------------
  m_rep_resid <- glmer(
    repeat. ~ pupil_resid + logRT_z + gain_centered + loss_centered + choice_prev1 +
      (1 + pupil_resid || sub),
    data = df_sub, family = binomial, control = ctrl
  )

  # ------------------------------------------------------------------
  # Test 3: RT × pupil interaction — effect stronger on slow trials?
  # ------------------------------------------------------------------
  m_rep_rtxpupil <- glmer(
    repeat. ~ pupil * logRT_z + gain_centered + loss_centered + choice_prev1 +
      (1 + pupil || sub),
    data = df_sub, family = binomial, control = ctrl
  )

  # ------------------------------------------------------------------
  # Test 4: Quadratic pupil — non-linear arousal effect?
  # ------------------------------------------------------------------
  m_rep_quad <- glmer(
    repeat. ~ pupil + I(pupil^2) + logRT_z + gain_centered + loss_centered + choice_prev1 +
      (1 + pupil || sub),
    data = df_sub, family = binomial, control = ctrl
  )

  models_named <- list(
    T2a_repBase    = m_rep_base,
    T2b_repResid   = m_rep_resid,
    T3_repRTxPupil = m_rep_rtxpupil,
    T4_repQuad     = m_rep_quad
  )

  diag <- lapply(models_named, check_model_diagnostics)
  for (nm in names(diag)) {
    message(sprintf("  %s — converged: %s, singular: %s",
                    nm, diag[[nm]]$converged, diag[[nm]]$singular))
  }

  robustness_models_list[[ds]] <- models_named
}

# =============================================================================
# Model comparison (T2a as reference — raw pupil baseline)
# =============================================================================

message("\nRunning robustness model comparison...")
rob_model_names <- c("T2a_repBase", "T2b_repResid", "T3_repRTxPupil", "T4_repQuad")

rob_comparison <- compare_models(
  model_list      = robustness_models_list,
  datasets        = datasets,
  models          = rob_model_names,
  reference_model = "T2a_repBase"
)

rob_diag_rows <- lapply(datasets, function(ds) {
  lapply(rob_model_names, function(nm) {
    d <- check_model_diagnostics(robustness_models_list[[ds]][[nm]])
    tibble(dataset = ds, model = nm,
           converged = d$converged, singular = d$singular)
  }) %>% bind_rows()
}) %>% bind_rows()

rob_comparison <- rob_comparison %>%
  left_join(rob_diag_rows, by = c("dataset", "model"))

write.csv(rob_comparison,
          file.path(analysis_results_path, "2_robustness_model_comparison.csv"),
          row.names = FALSE)
message("Saved: 2_robustness_model_comparison.csv")

# =============================================================================
# Fixed effects
# =============================================================================

message("Extracting robustness fixed effects...")
rob_fixed_effects <- extract_fixed_effects(
  model_list = robustness_models_list,
  datasets   = datasets,
  models     = rob_model_names,
  p_method   = p_method
)

rob_fixed_effects <- rob_fixed_effects %>%
  left_join(rob_diag_rows, by = c("dataset", "model"))

write.csv(rob_fixed_effects,
          file.path(analysis_results_path, "2_robustness_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_robustness_fixed_effects.csv")

message("\nAll pupil-behavior models complete.")
