# =============================================================================
# Pupil Betas — Transition x Mean Acceptance Model
# =============================================================================

formula_pupilbetas_transitionxmeanaccept <- pupil ~ logRT_z + gain_z + loss_z +
  entropy_c + abs_delta_ev_z + transition*mean_accept + (1 | sub)

# =============================================================================
# Fit per dataset (REML = TRUE -- this is the final reporting model)
# =============================================================================
transitionxmeanacceptmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting transition x mean acceptance model for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  pupilbetas_transitionxmeanaccept <- lmer(formula_pupilbetas_transitionxmeanaccept,
                                           data = df, REML = TRUE,
                                           control = LMER_CONTROL)
  print(summary(pupilbetas_transitionxmeanaccept)$varcor)

  transitionxmeanacceptmodel_list[[ds]] <- list(
    pupilbetas_transitionxmeanaccept = pupilbetas_transitionxmeanaccept
  )
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(transitionxmeanacceptmodel_list,
                                               datasets,
                                               "pupilbetas_transitionxmeanaccept")

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "4_pupilbetas_transitionxmeanaccept_convergence.csv"),
          row.names = FALSE)
message("Saved: 4_pupilbetas_transitionxmeanaccept_convergence.csv")

# =============================================================================
# Fixed effects
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = transitionxmeanacceptmodel_list,
  datasets   = datasets,
  models     = "pupilbetas_transitionxmeanaccept",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "4_pupilbetas_transitionxmeanaccept_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 4_pupilbetas_transitionxmeanaccept_fixed_effects.csv")

message("\nPupil transition x mean acceptance analysis complete.")
