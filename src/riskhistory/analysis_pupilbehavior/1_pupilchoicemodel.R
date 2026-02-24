# =============================================================================
# Pupil-Behavior GLME Model
# =============================================================================
#
# This model tests whether pupil size during stimulus presentation interacts
# with choice history (choice_prev1) to predict accept/reject decisions.
# =============================================================================

message("Fitting pupil-choice interaction models...")

for (ds in datasets) {
  message(sprintf("  Fitting model for dataset: %s", ds))

  # Subset data for this dataset
  df_sub <- data_clean[data_clean$dataset == ds, ]

  # Fit GLME model with continuous pupil size
  m <- glmer(
    accept ~ gain_centered * pupil_size_wc + loss_centered * pupil_size_wc +
      choice_prev1 * pupil_size_wc + rt +
      (1 + loss_centered | sub),
    data = df_sub,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 100000))
  )

  # Fit GLME model with pupil bins (low, medium, high)
  m_bin <- glmer(
    accept ~ gain_centered + loss_centered +
      choice_prev1 * pupil_bin + rt +
      (1 + loss_centered | sub) + (0 + choice_prev1 | sub) + (0 + pupil_bin | sub),
    data = df_sub,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 100000))
  )

  # Store model
  models_list[[ds]][["ModelPupilBehavior"]] <- m
  models_list[[ds]][["ModelPupilBehavior_Bin"]] <- m_bin

  message(sprintf("    Model fitted for %s", ds))
}

message("All models fitted successfully.")
