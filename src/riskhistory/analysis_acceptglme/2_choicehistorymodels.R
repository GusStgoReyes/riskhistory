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

for (ds in datasets) {
  print(paste("Running choice history models for dataset:", ds))
  df_sub <- data[data$dataset == ds, ]

  # CHOICE HISTORY MODEL (MODEL OF INTEREST)
  model2 <- glmer(accept ~ gain_centered + loss_centered + choice_prev1 +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  # PREVIOUS SUBJECTIVE VALUE MODEL
  model3 <- glmer(accept ~ gain_centered + loss_centered + prev_sv +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  # DELTA SUBJECTIVE VALUE MODEL
  model4 <- glmer(accept ~ gain_centered + loss_centered + delta_sv +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  # PREVIOUS STIMULI MODEL
  model5 <- glmer(accept ~ gain_centered + loss_centered + prev_gain_centered +
                    prev_loss_centered + (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  # PREVIOUS SUBJECTIVE VALUE MODEL + CHOICE HISTORY
  model6 <- glmer(accept ~ gain_centered + loss_centered + prev_sv +
                    choice_prev1 + (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  # DELTA SUBJECTIVE VALUE MODEL + CHOICE HISTORY
  model7 <- glmer(accept ~ gain_centered + loss_centered + delta_sv +
                    choice_prev1 + (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  # PREVIOUS STIMULI MODEL + CHOICE HISTORY
  model8 <- glmer(accept ~ gain_centered + loss_centered + prev_gain_centered +
                    prev_loss_centered + choice_prev1 +
                    (1 + loss_centered | sub),
                  data = df_sub, family = binomial)
  models <- list(
    M2 = model2,
    M3 = model3,
    M4 = model4,
    M5 = model5,
    M6 = model6,
    M7 = model7,
    M8 = model8
  )
  behavioral_models_list <- add_models_to_list(
    models_list = behavioral_models_list,
    ds     = ds,
    models = models
  )
}