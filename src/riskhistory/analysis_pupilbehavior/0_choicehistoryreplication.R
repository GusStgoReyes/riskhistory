# =============================================================================
# Choice History Under Risk: Replicate Choice-History Effect in Pupil Sample
# =============================================================================
# Before testing pupil interactions, corroborate -- within the pupil-eligible
# trial sample used by this analysis (data_clean), which is a subset of the
# full behavioral sample used in analysis_acceptglme -- that the choice-
# history model beats the baseline model, via AIC/BIC/BF model comparison.
#
#   M1: accept ~ gain_centered + loss_centered +
#                (1 + gain_centered + loss_centered | sub)
#   M2: accept ~ gain_centered + loss_centered + choice_prev1 +
#                (1 + gain_centered + loss_centered | sub)
# =============================================================================

choicehistoryreplication_models_list <- list()
for (ds in datasets) {
  choicehistoryreplication_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting choice-history replication models for dataset: %s", ds))
  df_sub <- data_clean %>% filter(dataset == ds)

  m1 <- glmer(
    accept ~ gain_centered + loss_centered +
      (1 + gain_centered + loss_centered | sub),
    data    = df_sub,
    family  = binomial,
    control = GLMER_CONTROL
  )
  diag1 <- check_model_diagnostics(m1)
  message(sprintf("  M1 converged: %s, singular: %s", diag1$converged, diag1$singular))

  m2 <- glmer(
    accept ~ gain_centered + loss_centered + choice_prev1 +
      (1 + gain_centered + loss_centered | sub),
    data    = df_sub,
    family  = binomial,
    control = GLMER_CONTROL
  )
  diag2 <- check_model_diagnostics(m2)
  message(sprintf("  M2 converged: %s, singular: %s", diag2$converged, diag2$singular))

  choicehistoryreplication_models_list[[ds]] <- list(M1 = m1, M2 = m2)
}

# =============================================================================
# Model comparison (AIC/BIC/BF), M2 vs M1
# =============================================================================

message("\nComparing M2 (choice history) vs M1 (baseline)...")
model_comparison <- compare_models(
  model_list      = choicehistoryreplication_models_list,
  datasets        = datasets,
  models          = c("M1", "M2"),
  reference_model = "M1"
)

write.csv(model_comparison,
          file.path(analysis_results_path, "0_choicehistoryreplication_model_comparison.csv"),
          row.names = FALSE)
message("Saved: 0_choicehistoryreplication_model_comparison.csv")

# =============================================================================
# Fixed effects (confirms choice_prev1 remains significant in this sample)
# =============================================================================

fixed_effects <- extract_fixed_effects(
  model_list = choicehistoryreplication_models_list,
  datasets   = datasets,
  models     = c("M1", "M2"),
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path, "0_choicehistoryreplication_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 0_choicehistoryreplication_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================

convergence_rows <- lapply(datasets, function(ds) {
  lapply(c("M1", "M2"), function(mdl) {
    d <- check_model_diagnostics(choicehistoryreplication_models_list[[ds]][[mdl]])
    data.frame(dataset = ds, model = mdl,
               converged = d$converged, singular = d$singular,
               stringsAsFactors = FALSE)
  })
})
convergence_df <- do.call(rbind, unlist(convergence_rows, recursive = FALSE))

write.csv(convergence_df,
          file.path(analysis_results_path, "0_choicehistoryreplication_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 0_choicehistoryreplication_convergence_report.csv")

message("\nChoice-history replication check complete.")
