# =============================================================================
# Choice History Under Risk: Run Pupil-Behavior Analysis
# =============================================================================
# This script runs the complete pupil-behavior GLME analysis pipeline:
# 1. Load and preprocess data (config)
# 2. Fit pupil-choice interaction models
# 3. Extract and save fixed effects with p-value correction
# =============================================================================

# 0. Load config file (loads data, applies exclusions, scales stimuli)
source("0_config.R")

# 1. Fit pupil-choice interaction models
source("1_pupilchoicemodel.R")

# =============================================================================
# Extract and save fixed effects
# =============================================================================
message("Extracting fixed effects...")

fixed_effects <- extract_all_fixed_effects(
  model_list = models_list,
  datasets   = datasets,
  p_method   = p_method
)

# Reorder columns for readability
fixed_effects <- fixed_effects %>%
  select(dataset, model, term, estimate, std.error, statistic,
         p.value, p_adj_bh, conf.low, conf.high)

# Save fixed effects
output_path <- file.path(analysis_results_path,
                         "1_pupilchoice_fixed_effects.csv")
write.csv(fixed_effects, output_path, row.names = FALSE)

message(sprintf("Fixed effects saved to: %s", output_path))

# =============================================================================
# Print summary
# =============================================================================
message("\n=== Analysis Summary ===")

model_names <- unique(fixed_effects$model)

for (ds in datasets) {
  message(sprintf("\nDataset: %s", ds))
  fe_ds <- fixed_effects[fixed_effects$dataset == ds, ]

  for (mn in model_names) {
    fe_mn <- fe_ds[fe_ds$model == mn, ]
    if (nrow(fe_mn) == 0) next
    message(sprintf("  Model: %s", mn))

    for (i in seq_len(nrow(fe_mn))) {
      term  <- fe_mn$term[i]
      est   <- round(fe_mn$estimate[i], 4)
      p_adj <- round(fe_mn$p_adj_bh[i], 4)
      sig   <- ifelse(p_adj < 0.001, "***",
                      ifelse(p_adj < 0.01, "**",
                             ifelse(p_adj < 0.05, "*", "")))
      message(sprintf("    %s: b = %.4f, p_adj = %.4f %s",
                      term, est, p_adj, sig))
    }
  }
}

# =============================================================================
# Convergence check
# =============================================================================
message("\nChecking model convergence...")

convergence_df <- check_convergence_all_models(
  model_list = models_list,
  datasets   = datasets
)

# Save convergence report
convergence_path <- file.path(analysis_results_path, "2_model_convergence_report.csv")
write.csv(convergence_df, convergence_path, row.names = FALSE)

n_models    <- nrow(convergence_df)
n_converged <- sum(convergence_df$converged)
n_failed    <- n_models - n_converged

message(sprintf("Convergence: %d / %d models converged.", n_converged, n_models))
if (n_failed > 0) {
  message(sprintf("WARNING: %d model(s) failed to converge:", n_failed))
  failed <- convergence_df[!convergence_df$converged, ]
  for (i in seq_len(nrow(failed))) {
    message(sprintf("  %s / %s", failed$dataset[i], failed$model[i]))
  }
}
message(sprintf("Convergence report saved to: %s", convergence_path))

message("\nAnalysis complete.")
