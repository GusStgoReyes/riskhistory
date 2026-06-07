# Runs sequential LRT (anova) across nested models for each dataset and
# returns a combined data frame with model names and dataset labels.
run_anova_comparison <- function(model_list, datasets, model_names) {
  results <- list()

  for (ds in datasets) {
    models <- model_list[[ds]][model_names]

    lrt    <- do.call(anova, unname(models))
    lrt_df <- as.data.frame(lrt)

    lrt_df$model   <- model_names
    lrt_df$dataset <- ds
    rownames(lrt_df) <- NULL

    results[[ds]] <- lrt_df
  }

  do.call(rbind, results)
}
