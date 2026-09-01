# =============================================================================
# Pupil Betas — Model Comparison (Choice Alteration vs. No Choice History)
# =============================================================================

source(file.path(shared_utils_path, "utils_modelcomparison.R"))

formula_pupilbetas_nohistory <- pupil ~ logRT_z + gain_z + loss_z +
  entropy_c + abs_delta_ev_z + accept + (0 + logRT_z | sub)

comparisonmodel_list <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  message(sprintf("\nFitting comparison models (ML) for dataset: %s", ds))
  df <- data_betas_clean %>% filter(dataset == ds)

  pupilbetas_choicealteration_ml <- lmer(
    formula_pupilbetas_choicealteration_continuous,
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  pupilbetas_nohistory_ml <- lmer(
    formula_pupilbetas_nohistory,
    data = df, REML = FALSE, control = LMER_CONTROL
  )

  comparisonmodel_list[[ds]] <- list(
    pupilbetas_choicealteration_ml = pupilbetas_choicealteration_ml,
    pupilbetas_nohistory_ml        = pupilbetas_nohistory_ml
  )
}

# =============================================================================
# Model comparison (AIC/BIC/log Bayes factor vs. the no-history baseline)
# =============================================================================
comparison_df <- compare_models(
  model_list      = comparisonmodel_list,
  datasets        = datasets,
  models          = c("pupilbetas_nohistory_ml", "pupilbetas_choicealteration_ml"),
  reference_model = "pupilbetas_nohistory_ml"
)

write.csv(comparison_df,
          file.path(analysis_results_path,
                    "4_pupilbetas_comparison.csv"),
          row.names = FALSE)
message("Saved: 4_pupilbetas_comparison.csv")

message("\nPupil betas model comparison complete.")
