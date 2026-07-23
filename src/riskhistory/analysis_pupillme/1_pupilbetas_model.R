# =============================================================================
# Choice History Under Risk: Pupil Betas -- Transition Model
# =============================================================================
# Same fixed-effects specification as analysis_pupiltimeseries's timepoint
# model, fit once per dataset (not per-timepoint) on the trial-level pupil
# GLM beta (decision-period amplitude, data_betas_clean$pupil) instead of the
# raw timeseries:
#
#   pupil ~ logRT_z + gain_z + loss_z + entropy_c + log_abs_delta_sv_z +
#           transition + (1 | sub)
#
# Replaces the previous streak-based, plain gain/loss, and 4-level-pattern
# pupil-beta models with a single model matching the transition
# operationalization used in analysis_pupiltimeseries.
# =============================================================================

acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

pupiltimeseries_path <- file.path(repo_path, "src", "riskhistory", "analysis_pupiltimeseries")
source(file.path(pupiltimeseries_path, "utils_createtransitionregressor.R"))

message("Adding transition regressor to pupil beta data...")
data_betas_clean <- add_transition_regressor(data_betas_clean)

pupilbetas_models_list <- list()
for (ds in datasets) {
  pupilbetas_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting pupil beta transition model for dataset: %s", ds))
  df_sub <- data_betas_clean %>% filter(dataset == ds)

  m <- lmer(
    pupil ~ logRT_z + gain_z + loss_z + entropy_c + log_abs_delta_sv_z +
      transition + (1 | sub),
    data    = df_sub,
    control = LMER_CONTROL
  )

  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  pupilbetas_models_list[[ds]] <- list(pupilbetas_transition = m)
}

# =============================================================================
# Fixed effects
# =============================================================================
message("\nExtracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = pupilbetas_models_list,
  datasets   = datasets,
  models     = "pupilbetas_transition",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path, "1_pupilbetas_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbetas_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================
convergence_rows <- lapply(datasets, function(ds) {
  d <- check_model_diagnostics(pupilbetas_models_list[[ds]][["pupilbetas_transition"]])
  data.frame(dataset = ds, model = "pupilbetas_transition",
             converged = d$converged, singular = d$singular,
             stringsAsFactors = FALSE)
})
convergence_df <- do.call(rbind, convergence_rows)

write.csv(convergence_df,
          file.path(analysis_results_path, "1_pupilbetas_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 1_pupilbetas_convergence_report.csv")

message("\nPupil beta transition model complete.")
