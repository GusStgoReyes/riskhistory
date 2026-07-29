# =============================================================================
# Utility functions for fitting timepoint models on pupil data
# =============================================================================

# Fit a single timepoint LME model
fit_timepoint_model <- function(data, formula, timebin, dataset, min_obs = 50,
                                emmeans_var = NULL, emmeans_contrasts = NULL) {
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
  # Estimated marginal means + specified contrasts for emmeans_var, via
  # marginaleffects::avg_predictions()/hypotheses() 
  # ---------------------------------------------------------------------
  emmeans_result <- tryCatch({
    pred <- avg_predictions(model, by = emmeans_var, re.form = NA)
    levels_in_order <- pred[[emmeans_var]]

    emm_df <- as.data.frame(pred) %>%
      rename(emmean = estimate, SE = std.error,
             asymp.LCL = conf.low, asymp.UCL = conf.high)
    emm_df$timebin <- timebin
    emm_df$dataset <- dataset

    contrasts_df <- NULL
    if (!is.null(emmeans_contrasts) && length(emmeans_contrasts) > 0) {
      group_expr <- function(group_levels) {
        idx <- match(group_levels, levels_in_order)
        paste0("(", paste(paste0("b", idx), collapse = " + "), ")/", length(idx))
      }

      hyp_strings <- character(0)
      hyp_kind    <- character(0)
      hyp_label   <- character(0)

      for (spec in emmeans_contrasts) {
        pos_expr <- group_expr(spec$pos)
        neg_expr <- group_expr(spec$neg)

        # A singleton group whose label is literally the raw level itself is
        # already present in emm_df via avg_predictions() -- echoing it again
        # here would just be a duplicate row. Only pooled groups (length > 1)
        # or singletons under a distinct custom label are genuinely new rows.
        pos_is_raw_level <- length(spec$pos) == 1 &&
          identical(spec$pos_label, spec$pos)
        neg_is_raw_level <- length(spec$neg) == 1 &&
          identical(spec$neg_label, spec$neg)

        hyp_strings <- c(hyp_strings,
                         paste0(pos_expr, " = 0"),
                         paste0(neg_expr, " = 0"),
                         paste0(pos_expr, " - ", neg_expr, " = 0"))
        hyp_kind  <- c(hyp_kind,
                       if (pos_is_raw_level) "group_skip" else "group",
                       if (neg_is_raw_level) "group_skip" else "group",
                       "contrast")
        hyp_label <- c(hyp_label, spec$pos_label, spec$neg_label, spec$name)
      }

      hyp_results <- as.data.frame(hypotheses(pred, hypothesis = hyp_strings)) %>%
        rename(SE = std.error, z.ratio = statistic)
      hyp_results$kind  <- hyp_kind
      hyp_results$label <- hyp_label

      group_rows <- hyp_results[hyp_results$kind == "group", ]
      if (nrow(group_rows) > 0) {
        extra_emm <- setNames(
          data.frame(group_rows$label, group_rows$estimate, group_rows$SE,
                    group_rows$conf.low, group_rows$conf.high, timebin, dataset,
                    stringsAsFactors = FALSE),
          c(emmeans_var, "emmean", "SE", "asymp.LCL", "asymp.UCL", "timebin", "dataset")
        )
        emm_df <- bind_rows(emm_df, extra_emm)
      }

      contrast_rows <- hyp_results[hyp_results$kind == "contrast", ]
      if (nrow(contrast_rows) > 0) {
        contrasts_df <- data.frame(
          contrast = contrast_rows$label,
          estimate = contrast_rows$estimate,
          SE       = contrast_rows$SE,
          z.ratio  = contrast_rows$z.ratio,
          p.value  = contrast_rows$p.value,
          stringsAsFactors = FALSE
        )
        contrasts_df$timebin <- timebin
        contrasts_df$dataset <- dataset
      }
    }

    list(emmeans = emm_df, contrasts = contrasts_df)
  }, error = function(e) {
    warning(sprintf("marginaleffects failed for timebin %.3f in %s: %s",
                    timebin, dataset, e$message))
    return(NULL)
  })

  list(
    fixed     = model_results,
    emmeans   = if (!is.null(emmeans_result)) emmeans_result$emmeans else NULL,
    contrasts = if (!is.null(emmeans_result)) emmeans_result$contrasts else NULL
  )
}


# Run timepoint models for a single dataset
run_models_for_dataset <- function(data, formula, dataset, min_obs = 50,
                                   emmeans_var = NULL, emmeans_contrasts = NULL) {
  timebins <- sort(unique(data$timebin))
  message(sprintf("    %d timebins, %d observations", length(timebins), nrow(data)))

  fixed_effects_list <- list()
  emmeans_list <- list()
  contrasts_list <- list()

  for (i in seq_along(timebins)) {
    tb      <- timebins[i]
    tb_data <- data %>% filter(timebin == tb)
    result  <- fit_timepoint_model(tb_data, formula, tb, dataset, min_obs,
                                   emmeans_var = emmeans_var,
                                   emmeans_contrasts = emmeans_contrasts)
    if (is.null(result)) next

    if (is.null(emmeans_var)) {
      fixed_effects_list[[length(fixed_effects_list) + 1]] <- result
    } else {
      fixed_effects_list[[length(fixed_effects_list) + 1]] <- result$fixed
      if (!is.null(result$emmeans)) {
        emmeans_list[[length(emmeans_list) + 1]] <- result$emmeans
      }
      if (!is.null(result$contrasts)) {
        contrasts_list[[length(contrasts_list) + 1]] <- result$contrasts
      }
    }
  }

  if (length(fixed_effects_list) == 0) return(NULL)

  list(
    fixed_effects = do.call(rbind, fixed_effects_list),
    emmeans       = if (length(emmeans_list) > 0) do.call(rbind, emmeans_list) else NULL,
    contrasts     = if (length(contrasts_list) > 0) do.call(rbind, contrasts_list) else NULL
  )
}


# Run timepoint models across all datasets with per-dataset p-value adjustment
run_all_timepoint_models <- function(data, formula, datasets, p_method = "BH",
                                     min_obs = 50, emmeans_var = NULL,
                                     emmeans_contrasts = NULL) {
  fixed_effects_all <- list()
  emmeans_all <- list()
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
                                         emmeans_contrasts = emmeans_contrasts)

    if (is.null(ds_results)) {
      warning(sprintf("No valid results for dataset %s", ds))
      next
    }

    ds_fixed <- ds_results$fixed_effects
    ds_fixed$p_adj_bh <- p.adjust(ds_fixed$p.value, method = p_method)
    fixed_effects_all[[length(fixed_effects_all) + 1]] <- ds_fixed

    if (!is.null(ds_results$emmeans)) {
      emmeans_all[[length(emmeans_all) + 1]] <- ds_results$emmeans
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
    emmeans       = if (length(emmeans_all) > 0) do.call(rbind, emmeans_all) else NULL,
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
