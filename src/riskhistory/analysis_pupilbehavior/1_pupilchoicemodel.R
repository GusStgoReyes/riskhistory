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
    accept ~ gain_centered + loss_centered +
      choice_prev1 * pupil +
      (1 + loss_centered | sub),
    data = df_sub,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 100000))
  )

  # Fit separate GLME models per pupil bin (low, medium, high)
  bins <- unique(df_sub$pupil_bin)
  for (bin in bins) {
    df_bin <- df_sub[df_sub$pupil_bin == bin, ]
    m_bin <- glmer(
      accept ~ gain_centered + loss_centered + choice_prev1 +
        (1 + loss_centered | sub),
      data = df_bin,
      family = binomial,
      control = glmerControl(optimizer = "bobyqa",
                             optCtrl = list(maxfun = 100000))
    )
    models_list[[ds]][[paste0("ModelPupilBehavior_Bin_", bin)]] <- m_bin
  }

  # Store model
  models_list[[ds]][["ModelPupilBehavior"]] <- m

  message(sprintf("    Model fitted for %s", ds))
}

message("All models fitted successfully.")
