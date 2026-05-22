# =============================================================================
# Choice History Under Risk: Loss Aversion Modulated by Tonic Arousal
# =============================================================================
# Tests whether loss sensitivity is modulated by tonic arousal (pre-trial
# pupil baseline: mean pupil amplitude in the 500 ms before trial onset,
# lowpass-filtered at 0.1 Hz) using a continuous interaction model:
#
#   accept ~ gain_sc + loss_sc * tonic_sc
#            + (1 + loss_sc + tonic_sc + gain_sc || sub)
# =============================================================================

if (!exists("behavioral_data")) source("0_config.R")

LA_GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# =============================================================================
# Load and merge tonic arousal (pre-trial baseline)
# =============================================================================

message("Loading pupil baseline data...")
pupilbaseline <- read.csv(file.path(analysis_data_path,
                                    "preprocessed",
                                    "pupilbaseline_preprocessed.csv")) %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned)) %>%
  select(sub, run, trial, dataset, exclude_startAligned, pupil_baseline)

data_tonic <- merge(behavioral_data, pupilbaseline,
                    by = c("sub", "run", "trial", "dataset")) %>%
  filter(!exclude_startAligned)

message(sprintf("Tonic arousal data: %d trials", nrow(data_tonic)))
for (ds in datasets) {
  n <- nrow(data_tonic[data_tonic$dataset == ds, ])
  message(sprintf("  %s: %d trials", ds, n))
}

# =============================================================================
# Fit model
# =============================================================================

lossaversion_models_list <- list()
for (ds in datasets) {
  lossaversion_models_list[[ds]] <- list()
}

la_convergence_rows <- list()

for (ds in datasets) {
  message(sprintf("\nFitting loss aversion × tonic arousal model for dataset: %s", ds))
  df_sub <- data_tonic %>%
    filter(dataset == ds) %>%
    mutate(
      gain_sc   = as.numeric(scale(gain_centered)),
      loss_sc   = as.numeric(scale(loss_centered)),
      tonic_sc  = as.numeric(scale(pupil_baseline))
    )

  m <- glmer(
    accept ~ gain_sc + loss_sc * tonic_sc +
      (1 + loss_sc + tonic_sc + gain_sc || sub),
    data    = df_sub,
    family  = binomial,
    control = LA_GLMER_CONTROL
  )

  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  lossaversion_models_list[[ds]] <- list(lossaversion = m)
  la_convergence_rows[[length(la_convergence_rows) + 1]] <- data.frame(
    dataset   = ds,
    converged = diag$converged,
    singular  = diag$singular,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Fixed effects
# =============================================================================

message("\nExtracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = lossaversion_models_list,
  datasets   = datasets,
  models     = "lossaversion",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path, "3_lossaversion_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_lossaversion_fixed_effects.csv")

# =============================================================================
# Convergence report
# =============================================================================

convergence_df <- do.call(rbind, la_convergence_rows)
write.csv(convergence_df,
          file.path(analysis_results_path, "3_lossaversion_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 3_lossaversion_convergence_report.csv")

message("\nLoss aversion × tonic arousal analysis complete.")
