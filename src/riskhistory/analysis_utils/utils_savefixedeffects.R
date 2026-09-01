extract_fixed_effects <- function(
  model_list,
  datasets,
  models,
  p_method = "BH"
) {

  fixed_effects <- list()
  i <- 1

  for (ds in datasets) {

    # Safety check
    if (!ds %in% names(model_list)) {
      warning(paste("Dataset", ds, "not found in model_list"))
      next
    }

    ds_models <- model_list[[ds]]

    for (name in models) {

      if (!name %in% names(ds_models)) {
        warning(paste("Model", name, "not found for dataset", ds))
        next
      }

      m <- ds_models[[name]]

      fe <- tidy(
        m,
        effects  = "fixed",
        conf.int = TRUE
      ) %>%
        mutate(
          dataset   = ds,
          model     = name,
          p_adj_bh  = p.adjust(p.value, method = p_method)
        )

      fixed_effects[[i]] <- fe
      i <- i + 1
    }
  }

  bind_rows(fixed_effects)
}