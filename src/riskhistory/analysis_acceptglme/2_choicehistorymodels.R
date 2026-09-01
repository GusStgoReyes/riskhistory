# =============================================================================
# Run the Confound Models in R
# =============================================================================
# Add subjective values and regressors to data (from previous file)
pth_betas <- file.path(analysis_results_path, "1_baselinebehavior_betas.csv")
data <- add_subjectivevalues_and_regressors(behavioral_data_full, pth_betas)
data <- subset(data, !exclude)

betas_model2 <- list()

for (ds in datasets) {
  print(paste("Running choice history models for dataset:", ds))
  df_sub <- data[data$dataset == ds, ]

  # CHOICE HISTORY MODEL (MODEL OF INTEREST)
  model2 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS EXPECTED VALUE MODEL
  model3 <- glmer(accept ~ gain_centered + loss_centered + prev_ev_z +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # DELTA EXPECTED VALUE MODEL
  model4 <- glmer(accept ~ gain_centered + loss_centered + delta_ev_z +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS GAIN MODEL
  model5 <- glmer(accept ~ gain_centered + loss_centered + prev_gain_z +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS LOSS MODEL
  model6 <- glmer(accept ~ gain_centered + loss_centered + prev_loss_z +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS EXPECTED VALUE MODEL + CHOICE HISTORY
  model7 <- glmer(accept ~ gain_centered + loss_centered + prev_ev_z +
                    choice_prev1 + (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # DELTA EXPECTED VALUE MODEL + CHOICE HISTORY
  model8 <- glmer(accept ~ gain_centered + loss_centered + delta_ev_z +
                    choice_prev1 + (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS GAIN + CHOICE HISTORY
  model9 <- glmer(accept ~ gain_centered + loss_centered + prev_gain_z +
                    choice_prev1 +
                    (1 + loss_centered + gain_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS LOSS + CHOICE HISTORY
  model10 <- glmer(accept ~ gain_centered + loss_centered + prev_loss_z +
                     choice_prev1 +
                     (1 + loss_centered + gain_centered | sub),
                   data = df_sub, family = binomial, control = GLMER_CONTROL)

  # CHOICE HISTORY MODEL + RND EFFECT OF CHOICE HISTORY
  if (ds == "equalRange") {
    model2_rnd <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                          (1 + loss_centered + gain_centered | sub),
                        data = df_sub, family = binomial, control = GLMER_CONTROL)
  } else {
    model2_rnd <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                          (1 + loss_centered + gain_centered  + choice_prev1 | sub),
                        data = df_sub, family = binomial, control = GLMER_CONTROL)
  }
  models <- list(
    M2_part1 = model2,
    M2_part1_rnd = model2_rnd,
    M3_part1 = model3,
    M4_part1 = model4,
    M5_part1 = model5,
    M6_part1 = model6,
    M7_part1 = model7,
    M8_part1 = model8,
    M9_part1 = model9,
    M10_part1 = model10
  )
  behavioral_models_list <- add_models_to_list(
    models_list = behavioral_models_list,
    ds     = ds,
    models = models
  )

}

# =============================================================================
# Model comparison and fixed effects for behavior models
# =============================================================================
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
write.csv(behavior_comparison_df, file.path(analysis_results_path,
                                            "2_behavior_model_comparison.csv"),
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
write.csv(fixed_effects, file.path(analysis_results_path,
                                   "2_behavior_fixed_effects.csv"),
          row.names = FALSE)

# =============================================================================
# Convergence check across all behavioral models
# =============================================================================
print("Checking convergence for all behavioral models...")
behavioral_models <- c("M1_part1", "M2_part1", "M2_part1_rnd", "M3_part1",
                       "M4_part1", "M5_part1", "M6_part1", "M7_part1",
                       "M8_part1", "M9_part1", "M10_part1")
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
