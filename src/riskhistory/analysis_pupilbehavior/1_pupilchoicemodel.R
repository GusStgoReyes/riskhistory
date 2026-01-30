# =============================================================================
# Pupil-Behavior GLME Model
# =============================================================================
# Model: accept ~ gain + loss + choice_prev1 * pupil_size +
#                 (1 + loss | sub) + (0 + choice_prev1 | sub)
#
# This model tests whether pupil size during stimulus presentation interacts
# with choice history (choice_prev1) to predict accept/reject decisions.
# =============================================================================

message("Fitting pupil-choice interaction models...")

for (ds in datasets) {
  message(sprintf("  Fitting model for dataset: %s", ds))

  # Subset data for this dataset
  df_sub <- data_clean[data_clean$dataset == ds, ]

  # Fit GLME model
  # accept ~ gain + loss + choice_prev1 * pupil_size +
  #          (1 + loss | sub) + (0 + choice_prev1 | sub)
  m <- glmer(
    accept ~ gain_centered + loss_centered +
             choice_prev1 * pupil_size +
             (1 + loss_centered | sub) + (0 + choice_prev1 | sub),
    data = df_sub,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 100000))
  )

  # Store model
  models_list[[ds]][["M1"]] <- m

  message(sprintf("    Model fitted for %s", ds))
}

message("All models fitted successfully.")
