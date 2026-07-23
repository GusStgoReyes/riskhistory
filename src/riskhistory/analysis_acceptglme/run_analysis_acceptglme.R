# =============================================================================
# Choice History Under Risk: Run the analysis for Choice History Effects in R
# =============================================================================
# 0. Load config file
source("0_config.R")

source("1_baselinechoicemodel.R")
source("2_choicehistorymodels.R")
source("4_supplementarymodels.R")


print("Extracting model comparison and fixed effects for behavior models...")
models <- c("M1_part1", "M2_part1", "M3_part1",
            "M4_part1", "M5_part1", "M6_part1",
            "M7_part1", "M8_part1", "M9_part1",
            "M10_part1")

behavior_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M2_part1"
)
# Save model comparison results
write.csv(behavior_comparison_df, file.path(analysis_results_path,
                                            "2_behavior_model_comparison_2.csv"),
          row.names = FALSE)

print("Extracting model comparison and fixed effects for behavior models...")
models <- c("M1_part1", "M2_part1_rnd")
behavior_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M1_part1"
)
# Save model comparison results
write.csv(behavior_comparison_df, file.path(analysis_results_path,
                                            "2_behavior_model_comparison_1.csv"),
          row.names = FALSE)
          
print("Extracting model comparison and fixed effects for behavior models...")
models <- c("M2_part1", "M2_part1_rnd")
behavior_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M2_part1"
)
# Save model comparison results
write.csv(behavior_comparison_df, file.path(analysis_results_path,
                                            "2_behavior_model_comparison_3.csv"),
          row.names = FALSE)
models <- c("M1_part1", "M2_part1", "M2_part1_rnd", "M3_part1",
            "M4_part1", "M5_part1", "M6_part1", "M7_part1",
            "M8_part1", "M9_part1", "M10_part1")
fixed_effects <- extract_fixed_effects(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = models,
  p_method   = p_method
)
# Save fixed effects results
write.csv(fixed_effects, file.path(analysis_results_path,
                                   "2_behavior_fixed_effects.csv"),
          row.names = FALSE)

print("Extracting model comparison and fixed effects for supplementary models...")
models <- c("M2_part1", "M_part1_Sup1")
supplementary_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M2_part1"
)
write.csv(supplementary_comparison_df,
          file.path(analysis_results_path,
                    "3_supplementary_model_comparison.csv"),
          row.names = FALSE)
supplementary_fixed_effects <- extract_fixed_effects(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = c("M_part1_Sup1"),
  p_method   = p_method
)
write.csv(supplementary_fixed_effects,
          file.path(analysis_results_path,
                    "3_supplementary_fixed_effects.csv"),
          row.names = FALSE)

print("Checking convergence for all behavioral models...")
behavioral_models <- c("M1_part1", "M2_part1", "M2_part1_rnd", "M3_part1", "M4_part1",
                       "M5_part1", "M6_part1", "M7_part1", "M8_part1", "M9_part1", "M10_part1", "M_part1_Sup1")
convergence_all <- check_convergence_all_models(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = behavioral_models
)

write.csv(convergence_all,
          file.path(analysis_results_path,
                    "4_model_convergence_report.csv"),
          row.names = FALSE)

n_models    <- nrow(convergence_all)
n_converged <- sum(convergence_all$converged)
n_failed    <- n_models - n_converged

print(paste0("Convergence check complete: ",
             n_converged, " of ", n_models,
             " models converged successfully."))
if (n_failed > 0) {
  print(paste0("WARNING: ", n_failed, " model(s) failed to converge."))
  print("See 4_model_convergence_report.csv for details.")
}

print("Analysis complete.")