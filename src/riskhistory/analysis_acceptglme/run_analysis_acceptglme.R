# =============================================================================
# Choice History Under Risk: Run the analysis for Choice History Effects in R
# =============================================================================
# 0. Load config file
source("0_config.R")

# 1. Baseline Model
source("1_baselinechoicemodel.R")
# 2. Choice History Models
source("2_choicehistorymodels.R")
# 3. Lag Choice History Models
source("3_lagchoicehistorymodels.R")
# 4. Response Time History Models
source("4_rthistorymodels.R")
# 5. Supplementary Models
source("5_supplementarymodels.R")

print("Extracting fixed effects for subjective value model...")
models <- c("M1_sv")
fixed_effects <- extract_fixed_effects(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = models,
  p_method   = p_method
)
# Save fixed effects results
write.csv(fixed_effects, file.path(analysis_results_path,
                                   "1_baselinebehavior_fixed_effects.csv"),
          row.names = FALSE)

print("Extracting model comparison and fixed effects for behavior models...")
models <- c("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
behavior_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M2"
)
# Save model comparison results
write.csv(behavior_comparison_df, file.path(analysis_results_path,
                                            "2_behavior_model_comparison.csv"),
          row.names = FALSE)
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

# Extracting the choice history random effect per subject

print(paste0("Extracting model comparison and fixed",
             "effects for lag choice history models..."))
models <- c("M1", "M9", "M10", "M11", "M12", "M13")
lagchoice_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M1"
)
# Save model comparison results
write.csv(lagchoice_comparison_df,
          file.path(analysis_results_path,
                    "3_lagchoice_model_comparison.csv"),
          row.names = FALSE)
lagchoice_fixed_effects <- extract_fixed_effects(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = c("M9", "M10", "M11", "M12", "M13"),
  p_method   = p_method
)
# Save fixed effects results
write.csv(lagchoice_fixed_effects,
          file.path(analysis_results_path,
                    "3_lagchoice_fixed_effects.csv"),
          row.names = FALSE)

print("Extracting fixed effects for RT models...")
models <- c("M_RT")
fixed_effects <- extract_fixed_effects(
  model_list = rt_models_list,
  datasets   = datasets,
  models     = models,
  p_method   = p_method
)
# Save fixed effects results
write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "4_rt_fixed_effects.csv"),
          row.names = FALSE)

print(paste0("Extracting model comparison and",
             "fixed effects for supplementary models..."))
models <- c("M2", "M_Sup1")
supplementary_comparison_df <- compare_models(
  model_list      = behavioral_models_list,
  datasets        = datasets,
  models          = models,
  reference_model = "M2"
)
# Save model comparison results
write.csv(supplementary_comparison_df,
          file.path(analysis_results_path,
                    "5_supplementary_model_comparison.csv"),
          row.names = FALSE)
supplementary_fixed_effects <- extract_fixed_effects(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = c("M_Sup1"),
  p_method   = p_method
)
# Save fixed effects results
write.csv(supplementary_fixed_effects,
          file.path(analysis_results_path,
                    "5_supplementary_fixed_effects.csv"),
          row.names = FALSE)

print("Checking convergence for all behavioral models...")
# Check convergence for all behavioral models
behavioral_models <- c("M1_sv", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
                       "M8", "M9", "M10", "M11", "M12", "M13", "M_Sup1")
convergence_behavioral <- check_convergence_all_models(
  model_list = behavioral_models_list,
  datasets   = datasets,
  models     = behavioral_models
)

# Check convergence for RT reject models
rt_models <- c("M_RT")
convergence_rt <- check_convergence_all_models(
  model_list = rt_models_list,
  datasets   = datasets,
  models     = rt_models
)

# Combine all convergence reports
convergence_all <- rbind(
  convergence_behavioral,
  convergence_rt
)

# Save convergence report
write.csv(convergence_all,
          file.path(analysis_results_path,
                    "6_model_convergence_report.csv"),
          row.names = FALSE)

# Print summary
n_models <- nrow(convergence_all)
n_converged <- sum(convergence_all$converged)
n_failed <- n_models - n_converged

print(paste0("Convergence check complete: ",
             n_converged, " of ", n_models,
             " models converged successfully."))
if (n_failed > 0) {
  print(paste0("WARNING: ", n_failed, " model(s) failed to converge."))
  print("See 6_model_convergence_report.csv for details.")
}

print("Analysis complete.")