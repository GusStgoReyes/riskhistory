# =============================================================================
# Choice History Under Risk: Pupil Beta LME Analysis
# =============================================================================
# Tests whether pupil dilation during the decision period (LS-A GLM beta) is
# associated with streak length at the moment of switching, over and above
# value-related and task variables.
#
# Models (fit per dataset):
#   M1     : pupil ~ sv + delta_sv + entropy + logRT + accept + (1 | sub)
#   M2     : M1 + logstreak_on_switch_c + (1 + logstreak_on_switch_c || sub)
# =============================================================================

library(bayestestR)

# Source model comparison and fixed effects utilities from analysis_acceptglme
acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_modelcomparison.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# =============================================================================
# Load and prepare pupil beta data
# =============================================================================

message("Loading pupil GLM betas...")
pupil_betas <- read.csv(file.path(analysis_data_path, "preprocessed",
                                  "pupilbetas_glm_preprocessed.csv"))
# Make exclude_startAligned boolean
pupil_betas <- pupil_betas %>%
  mutate(exclude_startAligned = as.logical(exclude_startAligned))

# Merge with behavioral data to obtain trial-level regressors
message("Merging with behavioral data...")
data_betas <- merge(pupil_betas,
                    behavioral_data,
                    by = c("sub", "trial", "run", "dataset"))
# Apply pupil exclusion criteria
data_betas_clean <- data_betas %>%
  filter(!exclude_startAligned)

# Scale regressors within-subject (reuses existing pupil regressor pipeline)
data_betas_clean <- create_pupil_regressors(data_betas_clean)

message(sprintf("Pupil beta data: %d trials after exclusion", nrow(data_betas_clean)))
for (ds in datasets) {
  n <- nrow(data_betas_clean[data_betas_clean$dataset == ds, ])
  message(sprintf("  %s: %d trials", ds, n))
}

# =============================================================================
# Fit models per dataset
# =============================================================================

pupilbeta_models_list <- list()
for (ds in datasets) {
  pupilbeta_models_list[[ds]] <- list()
}

for (ds in datasets) {
  message(sprintf("\nFitting pupil beta models for dataset: %s", ds))
  df_sub <- data_betas_clean %>% filter(dataset == ds)

  # M1: baseline — value, entropy, RT, accept (no streak)
  m1 <- lmer(pupil ~ sv_z + log_abs_delta_sv_z + entropy_c + logRT_z + accept +
               logstreak_on_repeat_z +
               (1 | sub) + (0 + accept | sub),
             data = df_sub, control = LMER_CONTROL)

  # M2: + switch streak
  m2 <- lmer(pupil ~ sv_z + log_abs_delta_sv_z + entropy_c + logRT_z + accept +
               logstreak_on_repeat_z + logstreak_on_switch_z +
               (1 | sub) + (0 + accept | sub) + (0 + logstreak_on_switch_z | sub),
             data = df_sub, control = LMER_CONTROL)

  diag <- lapply(list(M1_pupilbetas = m1, M2_pupilbetas = m2),
                 check_model_diagnostics)
  for (nm in names(diag)) {
    message(sprintf("  %s — converged: %s, singular: %s",
                    nm, diag[[nm]]$converged, diag[[nm]]$singular))
  }

  pupilbeta_models_list[[ds]] <- list(
    M1_pupilbetas = m1, M2_pupilbetas = m2
  )
}

# =============================================================================
# Save convergence / singularity report
# =============================================================================
convergence_rows <- lapply(datasets, function(ds) {
  lapply(names(pupilbeta_models_list[[ds]]), function(nm) {
    d <- check_model_diagnostics(pupilbeta_models_list[[ds]][[nm]])
    data.frame(dataset = ds, model = nm,
               converged = d$converged, singular = d$singular,
               stringsAsFactors = FALSE)
  })
})
convergence_pupilbetas <- do.call(rbind, unlist(convergence_rows, recursive = FALSE))

write.csv(convergence_pupilbetas,
          file.path(analysis_results_path, "3_pupilbetas_convergence_report.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_convergence_report.csv")

# =============================================================================
# Model comparison
# =============================================================================

message("\nRunning model comparison...")
model_comparison <- compare_models(
  model_list      = pupilbeta_models_list,
  datasets        = datasets,
  models          = c("M1_pupilbetas", "M2_pupilbetas"),
  reference_model = "M1_pupilbetas"
)

write.csv(model_comparison,
          file.path(analysis_results_path, "3_pupilbetas_model_comparison.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_model_comparison.csv")

# =============================================================================
# Fixed effects
# =============================================================================

message("Extracting fixed effects...")
fixed_effects <- extract_fixed_effects(
  model_list = pupilbeta_models_list,
  datasets   = datasets,
  models     = c("M1_pupilbetas", "M2_pupilbetas"),
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path, "3_pupilbetas_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_fixed_effects.csv")

message("\nPupil beta analysis complete.")

# =============================================================================
# Extract the random effect of logstreak_on_switch_z per subject and save
# for correlation with individual differences in choice history bias (beta_choice_prev)
# =============================================================================
message("Extracting per-subject random effects for logstreak_on_switch_z...")

randef_df <- lapply(names(pupilbeta_models_list), function(ds) {
  m2 <- pupilbeta_models_list[[ds]]$M2_pupilbetas
  re <- ranef(m2)$sub %>%
    as.data.frame() %>%
    tibble::rownames_to_column("sub")
  fe_switch <- fixef(m2)["logstreak_on_switch_z"]
  fe_accept <- fixef(m2)["accept"]
  re %>%
    transmute(
      sub,
      slope_logstreak = logstreak_on_switch_z + fe_switch,
      slope_accept = accept + fe_accept,
      dataset = ds
    )
}) %>%
  bind_rows()

write.csv(randef_df,
          file.path(analysis_results_path, "3_pupilbetas_random_effects_logstreak_on_switch.csv"),
          row.names = FALSE)
message("Saved: 3_pupilbetas_random_effects_logstreak_on_switch.csv")