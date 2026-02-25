# =============================================================================
# Utility: Extract fixed effects with p-value correction
# =============================================================================
# Extracts fixed effects from a model list and applies Benjamini-Hochberg
# p-value correction within each dataset.
# =============================================================================

extract_fixed_effects <- function(
  model_list,
  datasets,
  model_name = "M1",
  p_method = "BH"
) {

  fixed_effects_all <- list()
  i <- 1

  for (ds in datasets) {

    if (!ds %in% names(model_list)) {
      warning(paste("Dataset", ds, "not found in model_list"))
      next
    }

    m <- model_list[[ds]][[model_name]]

    if (is.null(m)) {
      warning(paste("Model", model_name, "not found for dataset", ds))
      next
    }

    fe <- tidy(
      m,
      effects  = "fixed",
      conf.int = TRUE
    )

    # Apply p-value correction within this dataset
    fe$p_adj_bh <- p.adjust(fe$p.value, method = p_method)

    # Add metadata
    fe$dataset <- ds
    fe$model <- model_name

    fixed_effects_all[[i]] <- fe
    i <- i + 1
  }

  bind_rows(fixed_effects_all)
}

# Extract fixed effects for all models present in model_list
extract_all_fixed_effects <- function(model_list, datasets, p_method = "BH") {

  model_names <- unique(unlist(lapply(datasets, function(ds) names(model_list[[ds]]))))

  results <- lapply(model_names, function(mn) {
    extract_fixed_effects(
      model_list = model_list,
      datasets   = datasets,
      model_name = mn,
      p_method   = p_method
    )
  })

  bind_rows(results)
}
