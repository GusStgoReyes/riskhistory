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

fixed_effects <- extract_fixed_effects(
  model_list = models_list,
  datasets   = datasets,
  model_name = "M1",
  p_method   = p_method
)

# Reorder columns for readability
fixed_effects <- fixed_effects %>%
  select(dataset, model, term, estimate, std.error, statistic,
         p.value, p_adj_bh, conf.low, conf.high)

# Save fixed effects
output_path <- file.path(analysis_results_path, "1_pupilchoice_fixed_effects.csv")
write.csv(fixed_effects, output_path, row.names = FALSE)

message(sprintf("Fixed effects saved to: %s", output_path))

# =============================================================================
# Print summary
# =============================================================================
message("\n=== Analysis Summary ===")

for (ds in datasets) {
  message(sprintf("\nDataset: %s", ds))
  fe_ds <- fixed_effects[fixed_effects$dataset == ds, ]

  for (i in seq_len(nrow(fe_ds))) {
    term <- fe_ds$term[i]
    est <- round(fe_ds$estimate[i], 4)
    p_adj <- round(fe_ds$p_adj_bh[i], 4)
    sig <- ifelse(p_adj < 0.001, "***",
                  ifelse(p_adj < 0.01, "**",
                         ifelse(p_adj < 0.05, "*", "")))
    message(sprintf("  %s: b = %.4f, p_adj = %.4f %s",
                    term, est, p_adj, sig))
  }
}

message("\nAnalysis complete.")
