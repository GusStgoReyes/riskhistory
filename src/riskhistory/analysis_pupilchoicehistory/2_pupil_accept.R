# =============================================================================
# Mean Acceptance ~ Pupil Beta by Transition
# =============================================================================
# Per dataset: does a subject's pupil beta in each of the 4 transition
# conditions (AA/AR/RR/RA) predict their overall mean acceptance rate, via
# lm(mean_accept_rate ~ pupilAA + pupilAR + pupilRR + pupilRA)?
# =============================================================================

message("Building per-subject pupil beta by transition...")
pupil_transition_df <- build_pupil_transition_df(data_betas_clean)

pupil_wide_df <- pupil_transition_df %>%
  pivot_wider(names_from = transition, values_from = pupil_beta,
              names_prefix = "pupil")

accept_cols <- c("mean_accept_rate", "pupilAA", "pupilAR", "pupilRR", "pupilRA")

model_data <- pupil_wide_df %>%
  inner_join(mean_accept_rate, by = c("sub", "dataset")) %>%
  filter(if_all(all_of(accept_cols), ~ !is.na(.)))

write.csv(model_data,
          file.path(analysis_results_path, "2_pupil_accept_data.csv"),
          row.names = FALSE)
message("Saved: 2_pupil_accept_data.csv")

# =============================================================================
# Fit one model per dataset
# =============================================================================
accept_models_list <- list()
fixed_effects_rows <- list()

for (ds in datasets) {
  message(sprintf("\nFitting mean_accept_rate ~ pupil betas, dataset: %s", ds))

  df_ds <- model_data %>% filter(dataset == ds)
  m <- lm(mean_accept_rate ~ pupilAA + pupilAR + pupilRR + pupilRA,
          data = df_ds)
  accept_models_list[[ds]] <- m

  message(sprintf("  n_subs=%d", nrow(df_ds)))

  fe <- broom::tidy(m, conf.int = TRUE) %>%
    mutate(dataset = ds, n_subs = nrow(df_ds))
  fixed_effects_rows[[length(fixed_effects_rows) + 1]] <- fe
}

fixed_effects_df <- bind_rows(fixed_effects_rows)

# BH-adjust each predictor's p-value across the 3 datasets
fixed_effects_df$p_adj_bh <- ave(
  fixed_effects_df$p.value,
  fixed_effects_df$term,
  FUN = function(p) p.adjust(p, method = p_method)
)

write.csv(fixed_effects_df,
          file.path(analysis_results_path, "2_pupil_accept_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 2_pupil_accept_fixed_effects.csv")

message("\nMean acceptance ~ pupil transition betas models complete.")
