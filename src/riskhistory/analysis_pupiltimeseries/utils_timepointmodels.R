# =============================================================================
# Utility functions for fitting timepoint models on pupil data
# =============================================================================

# Fit a single timepoint LME model
fit_timepoint_model <- function(data, formula, timebin, dataset, min_obs = 50,
                                emmeans_var = NULL, contrast_pairs = NULL) {
  if (nrow(data) < min_obs) {
    warning(sprintf("Insufficient data for timebin %.3f in %s (%d < %d), skipping...",
                    timebin, dataset, nrow(data), min_obs))
    return(NULL)
  }

  model <- tryCatch({
    lmer(formula, data = data, control = LMER_CONTROL)
  }, error = function(e) {
    warning(sprintf("Model failed for timebin %.3f in %s: %s", timebin, dataset, e$message))
    return(NULL)
  })

  if (is.null(model)) return(NULL)

  diagnostics <- check_model_diagnostics(model)

  model_results <- tryCatch({
    tidy(model, effects = "fixed", conf.int = TRUE, conf.level = 0.95)
  }, error = function(e) {
    warning(sprintf("Failed to extract results for timebin %.3f in %s: %s",
                    timebin, dataset, e$message))
    return(NULL)
  })

  if (is.null(model_results)) return(NULL)

  model_results$timebin   <- timebin
  model_results$dataset   <- dataset
  model_results$converged <- diagnostics$converged
  model_results$singular  <- diagnostics$singular

  if (is.null(emmeans_var)) {
    return(model_results)
  }

  # ---------------------------------------------------------------------
  # Average predictions + pairwise avg_comparisons()
  # ---------------------------------------------------------------------
  marginal_result <- tryCatch({
    pred <- avg_predictions(model, by = emmeans_var, re.form = NA)

    avgpred_df <- as.data.frame(pred) %>%
      rename(avgpred = estimate, SE = std.error,
             asymp.LCL = conf.low, asymp.UCL = conf.high)
    avgpred_df$timebin <- timebin
    avgpred_df$dataset <- dataset

    contrasts_df <- NULL
    if (!is.null(contrast_pairs) && length(contrast_pairs) > 0) {
      comp <- avg_comparisons(model,
                              variables = setNames(list("pairwise"), emmeans_var),
                              re.form = NA)
      comp_df <- as.data.frame(comp)

      # avg_comparisons()'s internal ordering for "pairwise" doesn't
      # necessarily match the order declared in contrast_pairs -- check
      # both "A - B" and "B - A" and normalize to the declared order,
      # flipping the sign of estimate/statistic when the reversed label
      # matched (p.value is sign-invariant, unaffected).
      matched_rows <- lapply(contrast_pairs, function(p) {
        label_fwd <- paste(p[1], "-", p[2])
        row_fwd <- comp_df[comp_df$contrast == label_fwd, ]
        if (nrow(row_fwd) > 0) {
          row_fwd$contrast <- label_fwd
          return(row_fwd)
        }
        label_rev <- paste(p[2], "-", p[1])
        row_rev <- comp_df[comp_df$contrast == label_rev, ]
        if (nrow(row_rev) > 0) {
          row_rev$contrast   <- label_fwd
          row_rev$estimate   <- -row_rev$estimate
          row_rev$statistic  <- -row_rev$statistic
          return(row_rev)
        }
        NULL
      })
      matched_rows <- matched_rows[!vapply(matched_rows, is.null, logical(1))]

      if (length(matched_rows) > 0) {
        comp_df <- do.call(rbind, matched_rows)
        contrasts_df <- data.frame(
          contrast = comp_df$contrast,
          estimate = comp_df$estimate,
          SE       = comp_df$std.error,
          z.ratio  = comp_df$statistic,
          p.value  = comp_df$p.value,
          stringsAsFactors = FALSE
        )
        contrasts_df$timebin <- timebin
        contrasts_df$dataset <- dataset
      }
    }

    list(avgpred = avgpred_df, contrasts = contrasts_df)
  }, error = function(e) {
    warning(sprintf("marginaleffects failed for timebin %.3f in %s: %s",
                    timebin, dataset, e$message))
    return(NULL)
  })

  list(
    fixed     = model_results,
    avgpred   = if (!is.null(marginal_result)) marginal_result$avgpred else NULL,
    contrasts = if (!is.null(marginal_result)) marginal_result$contrasts else NULL
  )
}


# Run timepoint models for a single dataset
run_models_for_dataset <- function(data, formula, dataset, min_obs = 50,
                                   emmeans_var = NULL, contrast_pairs = NULL) {
  timebins <- sort(unique(data$timebin))
  message(sprintf("    %d timebins, %d observations", length(timebins), nrow(data)))

  fixed_effects_list <- list()
  avgpred_list <- list()
  contrasts_list <- list()

  for (i in seq_along(timebins)) {
    tb      <- timebins[i]
    tb_data <- data %>% filter(timebin == tb)
    result  <- fit_timepoint_model(tb_data, formula, tb, dataset, min_obs,
                                   emmeans_var = emmeans_var,
                                   contrast_pairs = contrast_pairs)
    if (is.null(result)) next

    if (is.null(emmeans_var)) {
      fixed_effects_list[[length(fixed_effects_list) + 1]] <- result
    } else {
      fixed_effects_list[[length(fixed_effects_list) + 1]] <- result$fixed
      if (!is.null(result$avgpred)) {
        avgpred_list[[length(avgpred_list) + 1]] <- result$avgpred
      }
      if (!is.null(result$contrasts)) {
        contrasts_list[[length(contrasts_list) + 1]] <- result$contrasts
      }
    }
  }

  if (length(fixed_effects_list) == 0) return(NULL)

  list(
    fixed_effects = do.call(rbind, fixed_effects_list),
    avgpred       = if (length(avgpred_list) > 0) do.call(rbind, avgpred_list) else NULL,
    contrasts     = if (length(contrasts_list) > 0) do.call(rbind, contrasts_list) else NULL
  )
}


# Run timepoint models across all datasets with per-dataset p-value adjustment
run_all_timepoint_models <- function(data, formula, datasets, p_method = "BH",
                                     min_obs = 50, emmeans_var = NULL,
                                     contrast_pairs = NULL) {
  fixed_effects_all <- list()
  avgpred_all <- list()
  contrasts_all <- list()

  for (ds in datasets) {
    message(sprintf("  Processing dataset: %s", ds))
    ds_data <- data %>% filter(dataset == ds)

    if (nrow(ds_data) == 0) {
      warning(sprintf("No data for dataset %s, skipping...", ds))
      next
    }

    ds_results <- run_models_for_dataset(ds_data, formula, ds, min_obs,
                                         emmeans_var = emmeans_var,
                                         contrast_pairs = contrast_pairs)

    if (is.null(ds_results)) {
      warning(sprintf("No valid results for dataset %s", ds))
      next
    }

    ds_fixed <- ds_results$fixed_effects
    ds_fixed$p_adj_bh <- p.adjust(ds_fixed$p.value, method = p_method)
    fixed_effects_all[[length(fixed_effects_all) + 1]] <- ds_fixed

    if (!is.null(ds_results$avgpred)) {
      avgpred_all[[length(avgpred_all) + 1]] <- ds_results$avgpred
    }
    if (!is.null(ds_results$contrasts)) {
      ds_contrasts <- ds_results$contrasts
      ds_contrasts$p_adj_bh <- p.adjust(ds_contrasts$p.value, method = p_method)
      contrasts_all[[length(contrasts_all) + 1]] <- ds_contrasts
    }
  }

  if (length(fixed_effects_all) == 0) stop("No valid results from any dataset")

  list(
    fixed_effects = do.call(rbind, fixed_effects_all),
    avgpred       = if (length(avgpred_all) > 0) do.call(rbind, avgpred_all) else NULL,
    contrasts     = if (length(contrasts_all) > 0) do.call(rbind, contrasts_all) else NULL
  )
}


# Print summary of timepoint model results
print_results_summary <- function(results, datasets) {
  if (is.list(results) && "fixed_effects" %in% names(results)) {
    results <- results$fixed_effects
  }

  message("\n=== Summary ===")
  message(sprintf("Total results: %d rows", nrow(results)))

  for (ds in datasets) {
    ds_results <- results %>% filter(dataset == ds)
    if (nrow(ds_results) == 0) next
    n_timebins  <- length(unique(ds_results$timebin))
    n_terms     <- length(unique(ds_results$term))
    n_converged <- sum(ds_results$converged) / n_terms
    message(sprintf("  %s: %d timebins, %.0f%% converged",
                    ds, n_timebins, 100 * n_converged / n_timebins))
  }
}
