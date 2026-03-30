# =============================================================================
# Utility functions for fitting timepoint models on pupil data
# =============================================================================
# This file contains functions for fitting LME models at each timebin
# and collecting results with proper error handling.
# =============================================================================

#' Fit a single timepoint LME model
#'
#' @param data Data frame containing pupil data for a single timebin
#' @param formula Model formula to fit
#' @param timebin Numeric timebin value (for metadata)
#' @param dataset Character dataset name (for metadata)
#' @param min_obs Minimum observations required to fit model (default: 50)
#' @return List with fixed_effects and random_effects data frames, or NULL if fitting fails
fit_timepoint_model <- function(data, formula, timebin, dataset, min_obs = 50) {
  # Check for sufficient data
  if (nrow(data) < min_obs) {
    warning(sprintf("Insufficient data for timebin %.3f in %s (%d < %d), skipping...",
                    timebin, dataset, nrow(data), min_obs))
    return(NULL)
  }

  # Fit the model
  model <- tryCatch({
    lmer(formula,
         data = data,
         control = lmerControl(optimizer = "bobyqa",
                               optCtrl = list(maxfun = 100000)))
  }, error = function(e) {
    warning(sprintf("Model failed for timebin %.3f in %s: %s", timebin, dataset, e$message))
    return(NULL)
  })

  if (is.null(model)) return(NULL)

  # Check convergence/singularity
  diagnostics <- check_model_diagnostics(model)

  # Extract fixed effects with confidence intervals
  model_results <- tryCatch({
    tidy(model, effects = "fixed", conf.int = TRUE, conf.level = 0.95)
  }, error = function(e) {
    warning(sprintf("Failed to extract results for timebin %.3f in %s: %s",
                    timebin, dataset, e$message))
    return(NULL)
  })

  if (is.null(model_results)) return(NULL)

  # Add metadata
  model_results$timebin <- timebin
  model_results$dataset <- dataset
  model_results$converged <- diagnostics$converged
  model_results$singular <- diagnostics$singular

  return(list(
    fixed_effects = model_results))
}


#' Run timepoint models for a single dataset
#'
#' @param data Data frame containing pupil data for a single dataset
#' @param formula Model formula to fit
#' @param dataset Character dataset name
#' @param min_obs Minimum observations required to fit model (default: 50)
#' @return List with fixed_effects and random_effects data frames
run_models_for_dataset <- function(data, formula, dataset, min_obs = 50) {
  # Get timebins for this dataset
  timebins <- sort(unique(data$timebin))
  message(sprintf("    %d timebins, %d observations", length(timebins), nrow(data)))

  # Initialize results lists
  fixed_effects_list <- list()
  random_effects_list <- list()

  # Loop through timebins
  for (i in seq_along(timebins)) {
    tb <- timebins[i]

    # Get data for this timebin
    tb_data <- data %>%
      filter(timebin == tb)

    # Fit model
    tb_results <- fit_timepoint_model(tb_data, formula, tb, dataset, min_obs)

    if (!is.null(tb_results)) {
      if (!is.null(tb_results$fixed_effects)) {
        fixed_effects_list[[length(fixed_effects_list) + 1]] <- tb_results$fixed_effects
      }
    }
  }

  # Combine fixed effects results
  fixed_effects_df <- NULL
  if (length(fixed_effects_list) > 0) {
    fixed_effects_df <- do.call(rbind, fixed_effects_list)
  }

  # Combine random effects results
  random_effects_df <- NULL
  if (length(random_effects_list) > 0) {
    random_effects_df <- do.call(rbind, random_effects_list)
  }

  return(list(
    fixed_effects = fixed_effects_df,
    random_effects = random_effects_df
  ))
}


#' Run timepoint models across all datasets with per-dataset p-value adjustment
#'
#' @param data Full data frame containing all datasets
#' @param formula Model formula to fit
#' @param datasets Character vector of dataset names to process
#' @param p_method Method for p-value adjustment (default: "BH")
#' @param min_obs Minimum observations required to fit model (default: 50)
#' @return List with fixed_effects and random_effects data frames from all datasets
run_all_timepoint_models <- function(data, formula, datasets, p_method = "BH", min_obs = 50) {
  fixed_effects_all <- list()
  random_effects_all <- list()

  for (ds in datasets) {
    message(sprintf("  Processing dataset: %s", ds))

    # Get data for this dataset
    ds_data <- data %>%
      filter(dataset == ds)

    # Check if we have data
    if (nrow(ds_data) == 0) {
      warning(sprintf("No data for dataset %s, skipping...", ds))
      next
    }

    # Run models for this dataset
    ds_results <- run_models_for_dataset(ds_data, formula, ds, min_obs)

    if (is.null(ds_results$fixed_effects) && is.null(ds_results$random_effects)) {
      warning(sprintf("No valid results for dataset %s", ds))
      next
    }

    # Process fixed effects
    if (!is.null(ds_results$fixed_effects)) {
      # Apply p-value adjustment WITHIN this dataset
      # This adjusts across all estimates and timebins within the dataset
      ds_fixed <- ds_results$fixed_effects %>%
        mutate(p_adj_bh = p.adjust(p.value, method = p_method))
      fixed_effects_all[[length(fixed_effects_all) + 1]] <- ds_fixed
    }

    # Collect random effects
    if (!is.null(ds_results$random_effects)) {
      random_effects_all[[length(random_effects_all) + 1]] <- ds_results$random_effects
    }
  }

  # Combine all datasets
  if (length(fixed_effects_all) == 0) {
    stop("No valid results from any dataset")
  }

  fixed_effects_combined <- do.call(rbind, fixed_effects_all)
  random_effects_combined <- NULL
  if (length(random_effects_all) > 0) {
    random_effects_combined <- do.call(rbind, random_effects_all)
  }

  return(list(
    fixed_effects = fixed_effects_combined,
    random_effects = random_effects_combined
  ))
}


#' Print summary of timepoint model results
#'
#' @param results Data frame with fixed effects results (or list with fixed_effects element)
#' @param datasets Character vector of dataset names
print_results_summary <- function(results, datasets) {
  # Handle both old (data frame) and new (list) return formats
  if (is.list(results) && "fixed_effects" %in% names(results)) {
    results <- results$fixed_effects
  }

  message("\n=== Summary ===")
  message(sprintf("Total results: %d rows", nrow(results)))

  for (ds in datasets) {
    ds_results <- results %>% filter(dataset == ds)
    if (nrow(ds_results) == 0) next

    n_timebins <- length(unique(ds_results$timebin))
    n_terms <- length(unique(ds_results$term))
    n_converged <- sum(ds_results$converged) / n_terms
    message(sprintf("  %s: %d timebins, %.0f%% converged",
                    ds, n_timebins, 100 * n_converged / n_timebins))
  }
}