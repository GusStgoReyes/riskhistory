# =============================================================================
# Utility functions for fitting timepoint models on pupil data
# =============================================================================

#' Fit a single timepoint LME model
#'
#' @param data Data frame containing pupil data for a single timebin
#' @param formula Model formula to fit
#' @param timebin Numeric timebin value (for metadata)
#' @param dataset Character dataset name (for metadata)
#' @param min_obs Minimum observations required to fit model (default: 50)
#' @return Data frame of fixed effects, or NULL if fitting fails
fit_timepoint_model <- function(data, formula, timebin, dataset, min_obs = 50) {
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

  return(model_results)
}


#' Run timepoint models for a single dataset
#'
#' @param data Data frame containing pupil data for a single dataset
#' @param formula Model formula to fit
#' @param dataset Character dataset name
#' @param min_obs Minimum observations required to fit model (default: 50)
#' @return Data frame of fixed effects across all timebins
run_models_for_dataset <- function(data, formula, dataset, min_obs = 50) {
  timebins <- sort(unique(data$timebin))
  message(sprintf("    %d timebins, %d observations", length(timebins), nrow(data)))

  fixed_effects_list <- list()

  for (i in seq_along(timebins)) {
    tb      <- timebins[i]
    tb_data <- data %>% filter(timebin == tb)
    result  <- fit_timepoint_model(tb_data, formula, tb, dataset, min_obs)
    if (!is.null(result)) {
      fixed_effects_list[[length(fixed_effects_list) + 1]] <- result
    }
  }

  if (length(fixed_effects_list) == 0) return(NULL)
  do.call(rbind, fixed_effects_list)
}


#' Run timepoint models across all datasets with per-dataset p-value adjustment
#'
#' @param data Full data frame containing all datasets
#' @param formula Model formula to fit
#' @param datasets Character vector of dataset names to process
#' @param p_method Method for p-value adjustment (default: "BH")
#' @param min_obs Minimum observations required to fit model (default: 50)
#' @return Data frame of fixed effects from all datasets
run_all_timepoint_models <- function(data, formula, datasets, p_method = "BH", min_obs = 50) {
  fixed_effects_all <- list()

  for (ds in datasets) {
    message(sprintf("  Processing dataset: %s", ds))
    ds_data <- data %>% filter(dataset == ds)

    if (nrow(ds_data) == 0) {
      warning(sprintf("No data for dataset %s, skipping...", ds))
      next
    }

    ds_fixed <- run_models_for_dataset(ds_data, formula, ds, min_obs)

    if (is.null(ds_fixed)) {
      warning(sprintf("No valid results for dataset %s", ds))
      next
    }

    ds_fixed$p_adj_bh <- p.adjust(ds_fixed$p.value, method = p_method)
    fixed_effects_all[[length(fixed_effects_all) + 1]] <- ds_fixed
  }

  if (length(fixed_effects_all) == 0) stop("No valid results from any dataset")

  list(fixed_effects = do.call(rbind, fixed_effects_all))
}


#' Print summary of timepoint model results
#'
#' @param results List with fixed_effects element (or bare data frame)
#' @param datasets Character vector of dataset names
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
