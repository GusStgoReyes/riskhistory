# =============================================================================
# Choice History Under Risk: Pupil-Bin Behavioral Analysis
# =============================================================================
# Splits each subject's trials into low-, medium-, and high-arousal bins
# (within-subject tertile split on pupil), and fits one model per dataset
# where bin only moderates choice_prev1 (gain/loss are included as covariates
# but do not interact with bin):
#
#   accept ~ gain_centered + loss_centered + pupil_bin + choice_prev1 * pupil_bin +
#            (1 + loss_centered + gain_centered | sub)
#
# The choice_prev1:bin interaction term shows how arousal as indexed by pupil
# modulates the choice-history bias specifically.
# =============================================================================

pupilbins_models_list <- list()
for (ds in datasets) {
  pupilbins_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting pupil-bin model for dataset: %s", ds))
  df_sub <- data_clean %>% filter(dataset == ds)

  m <- glmer(
    accept ~ gain_centered + loss_centered + choice_prev1 * pupil_bin +
      (1 + loss_centered + gain_centered | sub),
    data    = df_sub,
    family  = binomial,
    control = GLMER_CONTROL
  )

  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  pupilbins_models_list[[ds]] <- list(pupilbins = m)
}

# =============================================================================
# Fixed effects
# =============================================================================

message("\nExtracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = pupilbins_models_list,
  datasets   = datasets,
  models     = "pupilbins",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path, "2_pupilbins_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbins_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================

convergence_rows <- lapply(datasets, function(ds) {
  d <- check_model_diagnostics(pupilbins_models_list[[ds]][["pupilbins"]])
  data.frame(dataset = ds, model = "pupilbins",
             converged = d$converged, singular = d$singular,
             stringsAsFactors = FALSE)
})
convergence_df <- do.call(rbind, convergence_rows)

write.csv(convergence_df,
          file.path(analysis_results_path, "2_pupilbins_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbins_convergence_report.csv")

message("\nPupil-bin analysis complete.")
