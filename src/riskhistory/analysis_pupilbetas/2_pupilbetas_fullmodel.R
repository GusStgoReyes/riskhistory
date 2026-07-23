# =============================================================================
# Pupil Betas — Full Model, Random Effects Fit Per Dataset
# =============================================================================
# Fixed effects are the same everywhere:
#   pupil ~ logRT_z + gain_z + loss_z + entropy_c + abs_delta_ev_z 
# =============================================================================

formulas_pupilbetas_full <- list(
  equalIndifference = pupil ~ logRT_z + gain_z + loss_z + entropy_c +
    abs_delta_ev_z +
    (1 | sub),

  equalRange = pupil ~ logRT_z + gain_z + loss_z + entropy_c +
    abs_delta_ev_z +
    (1 | sub),

  sheng2020 = pupil ~ logRT_z + gain_z + loss_z + entropy_c +
    abs_delta_ev_z +
    (1 | sub)
)

# =============================================================================
# Fit per dataset (REML = TRUE -- this is the final reporting model)
# =============================================================================
fullmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting full model for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  pupilbetas_full <- lmer(formulas_pupilbetas_full[[ds]],
                          data = df, REML = TRUE, control = LMER_CONTROL)
  print(summary(pupilbetas_full)$varcor)

  fullmodel_list[[ds]] <- list(pupilbetas_full = pupilbetas_full)
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(fullmodel_list, datasets,
                                               "pupilbetas_full")

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "2_pupilbetas_fullmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbetas_fullmodel_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = fullmodel_list,
  datasets   = datasets,
  models     = "pupilbetas_full",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "2_pupilbetas_fullmodel_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbetas_fullmodel_fixed_effects.csv")

# =============================================================================
# Residualized pupil (stage 1 of a two-stage transition analysis)
# =============================================================================
residuals_list <- list()
for (ds in datasets) {
  df <- data_betas_clean %>% filter(dataset == ds)
  m  <- fullmodel_list[[ds]][["pupilbetas_full"]]

  stopifnot(nrow(df) == length(residuals(m)))
  df$pupil_residual <- residuals(m)

  residuals_list[[ds]] <- df %>%
    select(sub, dataset, run, trial, pupil, pupil_residual)
}

pupil_residuals <- bind_rows(residuals_list)

write.csv(pupil_residuals,
          file.path(analysis_results_path,
                    "2_pupilbetas_fullmodel_residuals.csv"),
          row.names = FALSE)
message("Saved: 2_pupilbetas_fullmodel_residuals.csv")

# =============================================================================
# Append pupil_residual to the preprocessed pupil betas file, so downstream
# consumers that expect that file's shape (e.g. analysis_ddm's --pupil_pth,
# which relies on exclude_startAligned) get pupil_residual too. Excluded rows
# (not part of data_betas_clean) get pupil_residual = NA.
# =============================================================================
pupilbetas_preprocessed_path <- file.path(analysis_data_path, "preprocessed",
                                          "pupilbetas_glm_preprocessed.csv")
pupilbetas_preprocessed <- read.csv(pupilbetas_preprocessed_path)

pupilbetas_preprocessed <- pupilbetas_preprocessed %>%
  select(-any_of("pupil_residual")) %>%
  left_join(
    pupil_residuals %>% select(sub, dataset, run, trial, pupil_residual),
    by = c("sub", "dataset", "run", "trial")
  )

write.csv(pupilbetas_preprocessed, pupilbetas_preprocessed_path,
          row.names = FALSE)
message("Appended pupil_residual to: ",
        "data/preprocessed/pupilbetas_glm_preprocessed.csv")
