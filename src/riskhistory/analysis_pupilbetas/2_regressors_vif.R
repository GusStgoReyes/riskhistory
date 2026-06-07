# =============================================================================
# Pupil Betas — VIF for M6 regressors
# =============================================================================
library(car)

vif_results <- list()

for (ds in datasets) {
  message(sprintf("Computing VIF for dataset: %s", ds))

  m6  <- models_list[[ds]]$M6_switch_accept
  vif_out <- car::vif(m6)

  # vif() returns a matrix with GVIF, Df, GVIF^(1/(2*Df)) when interactions
  # are present, and a named vector otherwise.
  if (is.matrix(vif_out)) {
    vif_df <- as.data.frame(vif_out)
    vif_df$term <- rownames(vif_df)
    rownames(vif_df) <- NULL
  } else {
    vif_df <- data.frame(term = names(vif_out), GVIF = vif_out,
                         row.names = NULL)
  }

  vif_df$dataset <- ds
  vif_results[[ds]] <- vif_df
}

vif_all <- do.call(rbind, vif_results)
rownames(vif_all) <- NULL

write.csv(vif_all,
          file.path(analysis_results_path, "2_regressors_vif.csv"),
          row.names = FALSE)
message("Saved: 2_regressors_vif.csv")

# =============================================================================
# Correlation matrix of M6 regressors
# =============================================================================
# Add interaction term of accept:switch to the correlation analysis

regressor_cols <- c("logRT_z", "gain_z", "loss_z", "entropy_c",
                    "log_abs_delta_sv_z", "accept", "switch", "accept_switch")

cor_results <- list()

for (ds in datasets) {
  df <- data_betas_clean %>%
    filter(dataset == ds) %>%
    select(all_of(regressor_cols))

  cor_mat <- cor(df, use = "pairwise.complete.obs")

  cor_long <- as.data.frame(as.table(cor_mat))
  colnames(cor_long) <- c("var1", "var2", "correlation")
  cor_long$dataset <- ds
  cor_results[[ds]] <- cor_long
}

cor_all <- do.call(rbind, cor_results)
rownames(cor_all) <- NULL

write.csv(cor_all,
          file.path(analysis_results_path, "2_regressors_correlation.csv"),
          row.names = FALSE)
message("Saved: 2_regressors_correlation.csv")
