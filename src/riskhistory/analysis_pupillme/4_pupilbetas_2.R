# =============================================================================
# Choice History Under Risk: Pupil Beta LME Analysis — Model Building
# =============================================================================
# Incrementally builds up the pupil beta model starting from gain/loss.
# gain_centered / loss_centered are mean-centred within the full sample.
# =============================================================================

# =============================================================================
# Load and prepare pupil beta data (if not already loaded by 3_)
# =============================================================================

if (!exists("data_betas_clean")) {
  message("Loading pupil GLM betas...")
  pupil_betas <- read.csv(
    file.path(analysis_data_path, "preprocessed",
              "pupilbetas_glm_preprocessed.csv")
  )
  pupil_betas <- pupil_betas %>%
    mutate(exclude_startAligned = as.logical(exclude_startAligned))

  message("Merging with behavioral data...")
  data_betas <- merge(pupil_betas, behavioral_data,
                      by = c("sub", "trial", "run", "dataset"))
  data_betas_clean <- data_betas %>%
    filter(!exclude_startAligned)
  data_betas_clean <- create_pupil_regressors(data_betas_clean)

  message(sprintf("Pupil beta data: %d trials after exclusion",
                  nrow(data_betas_clean)))
}

# =============================================================================
# Fit models per dataset
# =============================================================================

pupilbeta2_models_list <- list()
for (ds in datasets) {
  pupilbeta2_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting pupil beta-2 models for dataset: %s", ds))
  df_sub <- data_betas_clean %>% filter(dataset == ds)

  m1_pb2 <- lmer(
    pupil ~ gain_centered + loss_centered + logRT_z +
      (1 | sub) +
      (0 + gain_centered | sub) +
      (0 + loss_centered | sub) +
      (0 + logRT_z | sub),
    data = df_sub, control = LMER_CONTROL
  )
  print(summary(m1_pb2))

  diag <- check_model_diagnostics(m1_pb2)
  message(sprintf("  M1_pb2 — converged: %s, singular: %s",
                  diag$converged, diag$singular))

  pupilbeta2_models_list[[ds]] <- list(M1_pb2 = m1_pb2)
}

# =============================================================================
# Save convergence report
# =============================================================================

convergence_rows_pb2 <- lapply(datasets, function(ds) {
  lapply(names(pupilbeta2_models_list[[ds]]), function(nm) {
    d <- check_model_diagnostics(pupilbeta2_models_list[[ds]][[nm]])
    data.frame(dataset = ds, model = nm,
               converged = d$converged, singular = d$singular,
               stringsAsFactors = FALSE)
  })
})
convergence_pb2 <- do.call(
  rbind, unlist(convergence_rows_pb2, recursive = FALSE)
)

write.csv(convergence_pb2,
          file.path(analysis_results_path,
                    "4_pupilbetas2_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 4_pupilbetas2_convergence_report.csv")

# =============================================================================
# Fixed effects
# =============================================================================

message("Extracting fixed effects...")
fixed_effects_pb2 <- extract_fixed_effects(
  model_list = pupilbeta2_models_list,
  datasets   = datasets,
  models     = c("M1_pb2"),
  p_method   = p_method
)

write.csv(fixed_effects_pb2,
          file.path(analysis_results_path,
                    "4_pupilbetas2_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 4_pupilbetas2_fixed_effects.csv")

message("\nPupil beta-2 model building complete.")
