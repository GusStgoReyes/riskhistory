# =============================================================================
# Choice History Under Risk: Run the Behavior Models with History in R
# =============================================================================
# Run several behavioral models including choice history and stimuli history for
# each dataset and extract fixed effects.
pth_betas <- file.path(analysis_results_path, "1_baselinebehavior_betas.csv")
data <- add_subjectivevalues(
  df = behavioral_data_full,
  pth_betas = pth_betas
)
data <- create_behavioral_regressors(data)

betas_model2 <- list()

for (ds in datasets) {
  print(paste("Running choice history models for dataset:", ds))
  df_sub <- data[data$dataset == ds, ]

  # CHOICE HISTORY MODEL (MODEL OF INTEREST)
  model2 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS SUBJECTIVE VALUE MODEL
  model3 <- glmer(accept ~ gain_centered + loss_centered + prev_sv_z +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # DELTA SUBJECTIVE VALUE MODEL
  model4 <- glmer(accept ~ gain_centered + loss_centered + delta_sv_z +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS STIMULI MODEL
  model5 <- glmer(accept ~ gain_centered + loss_centered + prev_gain_centered +
                    prev_loss_centered + (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS SUBJECTIVE VALUE MODEL + CHOICE HISTORY
  model6 <- glmer(accept ~ gain_centered + loss_centered + prev_sv_z +
                    choice_prev1 + (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # DELTA SUBJECTIVE VALUE MODEL + CHOICE HISTORY
  model7 <- glmer(accept ~ gain_centered + loss_centered + delta_sv_z +
                    choice_prev1 + (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)
  # PREVIOUS STIMULI MODEL + CHOICE HISTORY
  model8 <- glmer(accept ~ gain_centered + loss_centered + prev_gain_centered +
                    prev_loss_centered + choice_prev1 +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial, control = GLMER_CONTROL)

  # CHOICE HISTORY MODEL + RND EFFECT OF CHOICE HISTORY
  model2_rnd <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                        (1 + loss_centered | sub) + (0 + choice_prev1 | sub),
                      data = df_sub, family = binomial, control = GLMER_CONTROL)
  models <- list(
    M2_part1 = model2,
    M2_part1_rnd = model2_rnd,
    M3_part1 = model3,
    M4_part1 = model4,
    M5_part1 = model5,
    M6_part1 = model6,
    M7_part1 = model7,
    M8_part1 = model8
  )
  behavioral_models_list <- add_models_to_list(
    models_list = behavioral_models_list,
    ds     = ds,
    models = models
  )

  # COMPUTE CHOICE HISTORY RANDOM EFFECTS PER SUBJECT
  betas_model2[[ds]] <- extract_random_effects_history(
    model = model2_rnd,
    ds = ds
  )
}

# Combine all datasets' betas into a single data frame
final_betas_model2 <- bind_rows(betas_model2)

# Save to CSV
write.csv(final_betas_model2, file.path(analysis_results_path,
                                        "2_choicehistory_betas.csv"),
          row.names = FALSE)