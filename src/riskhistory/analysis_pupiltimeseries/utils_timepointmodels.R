# =============================================================================
# Utility functions for fitting timepoint models on pupil data
# =============================================================================
# normalize_pairwise_contrasts() lives in analysis_utils/utils_contrasts.R

# Equal-weight means of groups of `by_var` levels (e.g. accept = mean of A-A
# and R-A), plus the first-minus-second group contrast
pooled_predictions <- function(pred, by_var, pooled_levels) {
  pred_levels <- as.character(as.data.frame(pred)[[by_var]])
  weights <- sapply(pooled_levels,
                    function(lv) (pred_levels %in% lv) / length(lv))

  contrast_weights <- weights[, 1, drop = FALSE] - weights[, 2]
  colnames(contrast_weights) <- paste(names(pooled_levels)[1:2],
                                      collapse = " - ")

  list(
    avgpred_choice = hypotheses(pred, hypothesis = weights) %>%
      as.data.frame() %>%
      transmute(choice = term, avgpred = estimate, SE = std.error,
                statistic, p.value, asymp.LCL = conf.low,
                asymp.UCL = conf.high),
    contrasts_choice = hypotheses(pred, hypothesis = contrast_weights) %>%
      as.data.frame() %>%
      transmute(contrast = term, estimate, SE = std.error,
                z.ratio = statistic, p.value)
  )
}

# Fit a single timepoint LME model; returns a named list of result tables
fit_timepoint_model <- function(data, formula, timebin, dataset, min_obs = 50,
                                emmeans_var = NULL, contrast_pairs = NULL,
                                pooled_levels = NULL) {
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

  model_results$converged <- diagnostics$converged
  model_results$singular  <- diagnostics$singular
  tables <- list(fixed_effects = model_results)

  # ---------------------------------------------------------------------
  # Average predictions, pairwise contrasts, and pooled-level means
  # ---------------------------------------------------------------------
  if (!is.null(emmeans_var)) {
    marginal_tables <- tryCatch({
      pred <- avg_predictions(model, by = emmeans_var, re.form = NA)
      out <- list(
        avgpred = as.data.frame(pred) %>%
          rename(avgpred = estimate, SE = std.error,
                 asymp.LCL = conf.low, asymp.UCL = conf.high)
      )

      if (length(contrast_pairs) > 0) {
        comp <- avg_comparisons(model,
                                variables = setNames(list("pairwise"), emmeans_var),
                                re.form = NA)
        out$contrasts <- normalize_pairwise_contrasts(as.data.frame(comp),
                                                      contrast_pairs)
      }

      if (length(pooled_levels) > 0) {
        out <- c(out, pooled_predictions(pred, emmeans_var, pooled_levels))
      }
      out
    }, error = function(e) {
      warning(sprintf("marginaleffects failed for timebin %.3f in %s: %s",
                      timebin, dataset, e$message))
      list()
    })
    tables <- c(tables, marginal_tables)
  }

  tables <- tables[!vapply(tables, is.null, logical(1))]
  lapply(tables, function(df) {
    df$timebin <- timebin
    df$dataset <- dataset
    df
  })
}


# Row-bind each named table across a list of per-timebin/per-dataset results
bind_result_tables <- function(results) {
  results <- results[!vapply(results, is.null, logical(1))]
  table_names <- unique(unlist(lapply(results, names)))
  setNames(lapply(table_names, function(name) {
    bind_rows(lapply(results, `[[`, name))
  }), table_names)
}


# Run timepoint models for a single dataset
run_models_for_dataset <- function(data, formula, dataset, min_obs = 50,
                                   emmeans_var = NULL, contrast_pairs = NULL,
                                   pooled_levels = NULL) {
  timebins <- sort(unique(data$timebin))
  message(sprintf("    %d timebins, %d observations", length(timebins), nrow(data)))

  results <- lapply(timebins, function(tb) {
    fit_timepoint_model(data %>% filter(timebin == tb), formula, tb, dataset,
                        min_obs, emmeans_var = emmeans_var,
                        contrast_pairs = contrast_pairs,
                        pooled_levels = pooled_levels)
  })
  if (all(vapply(results, is.null, logical(1)))) return(NULL)

  bind_result_tables(results)
}


# Run timepoint models across all datasets, BH-adjusting fixed effects and
# contrasts across timebins within each dataset
run_all_timepoint_models <- function(data, formula, datasets, p_method = "BH",
                                     min_obs = 50, emmeans_var = NULL,
                                     contrast_pairs = NULL,
                                     pooled_levels = NULL) {
  adjusted_tables <- c("fixed_effects", "contrasts", "contrasts_choice")

  results <- lapply(datasets, function(ds) {
    message(sprintf("  Processing dataset: %s", ds))
    ds_data <- data %>% filter(dataset == ds)
    if (nrow(ds_data) == 0) {
      warning(sprintf("No data for dataset %s, skipping...", ds))
      return(NULL)
    }

    ds_results <- run_models_for_dataset(ds_data, formula, ds, min_obs,
                                         emmeans_var = emmeans_var,
                                         contrast_pairs = contrast_pairs,
                                         pooled_levels = pooled_levels)
    if (is.null(ds_results)) {
      warning(sprintf("No valid results for dataset %s", ds))
      return(NULL)
    }

    for (name in intersect(adjusted_tables, names(ds_results))) {
      ds_results[[name]]$p_adj_bh <- p.adjust(ds_results[[name]]$p.value,
                                              method = p_method)
    }
    ds_results
  })

  if (all(vapply(results, is.null, logical(1)))) {
    stop("No valid results from any dataset")
  }
  bind_result_tables(results)
}
