# =============================================================================
# Choice History Under Risk: Pupil-Bin x Choice History Simple Effects
# =============================================================================
# Uses the single per-dataset model fit in 2_pupilbins_models.R
#
#   accept ~ gain_centered + loss_centered + choice_prev1 * pupil_bin +
#            (1 + loss_centered + gain_centered | sub)
#
# to compute, via emmeans (which correctly propagates the fixed-effect
# covariance, unlike summing independent SEs off the raw coefficient table),
# the predicted P(accept) at choice_prev1 = -1 vs +1, crossed with pupil_bin
# (Low/Medium/High) -- this is what gets visualized to show how the
# choice_prev1 bias plays out within each arousal bin -- plus the pairwise
# contrast between the two choice_prev1 levels within each bin (odds ratio,
# with a Wald p-value on the log-odds scale).
# =============================================================================

choiceprev1_predicted_rows <- list()
choiceprev1_contrast_rows  <- list()

for (ds in datasets) {
  message(sprintf(
    "\nComputing choice_prev1 x pupil_bin predicted P(accept) for dataset: %s",
    ds
  ))
  m <- pupilbins_models_list[[ds]][["pupilbins"]]

  emm <- emmeans(m, ~ choice_prev1 * pupil_bin,
                 at = list(choice_prev1 = c(-1, 1)),
                 type = "response")

  choiceprev1_predicted_rows[[ds]] <- as.data.frame(emm) %>%
    mutate(dataset = ds) %>%
    select(dataset, choice_prev1, pupil_bin, prob, SE, df, asymp.LCL, asymp.UCL)

  choiceprev1_contrast_rows[[ds]] <- as.data.frame(
    contrast(emm, method = "revpairwise", by = "pupil_bin", type = "response")
  ) %>%
    mutate(dataset = ds) %>%
    select(dataset, pupil_bin, contrast, odds.ratio, SE, df, z.ratio, p.value)
}

choiceprev1_predicted_df <- bind_rows(choiceprev1_predicted_rows)
choiceprev1_contrast_df  <- bind_rows(choiceprev1_contrast_rows)

write.csv(
  choiceprev1_predicted_df,
  file.path(analysis_results_path, "2b_pupilbins_choiceprev1_predicted.csv"),
  row.names = FALSE
)
message("Saved: 2b_pupilbins_choiceprev1_predicted.csv")

write.csv(
  choiceprev1_contrast_df,
  file.path(analysis_results_path, "2b_pupilbins_choiceprev1_contrasts.csv"),
  row.names = FALSE
)
message("Saved: 2b_pupilbins_choiceprev1_contrasts.csv")

message("\nPupil-bin simple effects complete.")
