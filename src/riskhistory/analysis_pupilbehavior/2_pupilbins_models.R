# =============================================================================
# Choice History Under Risk: Pupil-Bin Behavioral Analysis
# =============================================================================
# Splits each subject's trials into low-, medium-, and high-arousal bins
# (within-subject tertile split on pupil) and fits a separate GLME in each bin:
#
#   accept ~ gain_centered + loss_centered + choice_prev1 + (1 + loss_centered | sub)
#
# Comparing fixed effects between bins shows how arousal as indexed by pupil
# modulates sensitivity to value and choice history.
# =============================================================================

pupilbins_models_list <- list()
for (ds in datasets) {
  pupilbins_models_list[[ds]] <- list()
}

BINS <- c("q1", "q2", "q3", "q4")

for (ds in datasets) {
  message(sprintf("\nFitting pupil-bin models for dataset: %s", ds))

  for (bin in BINS) {
    message(sprintf("  bin: %s", bin))
    df_bin <- data_clean %>% filter(dataset == ds, pupil_bin == bin)

    m <- glmer(
      accept ~ gain_centered + loss_centered + choice_prev1 + (1 + loss_centered | sub),
      data    = df_bin,
      family  = binomial,
      control = GLMER_CONTROL
    )

    diag <- check_model_diagnostics(m)
    message(sprintf("    converged: %s, singular: %s", diag$converged, diag$singular))

    pupilbins_models_list[[ds]][[bin]] <- m
  }
}

# =============================================================================
# Fixed effects
# =============================================================================

message("\nExtracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = pupilbins_models_list,
  datasets   = datasets,
  models     = BINS,
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
  lapply(BINS, function(bin) {
    d <- check_model_diagnostics(pupilbins_models_list[[ds]][[bin]])
    data.frame(dataset = ds, pupil_bin = bin,
               converged = d$converged, singular = d$singular,
               stringsAsFactors = FALSE)
  })
})
convergence_df <- do.call(rbind, unlist(convergence_rows, recursive = FALSE))

write.csv(convergence_df,
          file.path(analysis_results_path, "2_pupilbins_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbins_convergence_report.csv")

message("\nPupil-bin analysis complete.")
