compare_models <- function(
  model_list,
  datasets,
  models,
  reference_model
) {

  model_metrics <- list()
  i <- 1
  for (ds in datasets) {

    # Safety check
    if (!ds %in% names(model_list)) {
      warning(paste("Dataset", ds, "not found in model_list"))
      next
    }

    ds_models <- model_list[[ds]]

    if (!reference_model %in% names(ds_models)) {
      warning(paste("Reference model", reference_model,
                    "not found for dataset", ds))
      next
    }

    ref_mod <- ds_models[[reference_model]]

    for (mdl in models) {
      print(paste("Comparing model", mdl, "for dataset", ds))
      if (!mdl %in% names(ds_models)) {
        warning(paste("Model", mdl, "not found for dataset", ds))
        next
      }
      mod <- ds_models[[mdl]]
      model_metrics[[i]] <- tibble(
        dataset = ds,
        model   = mdl,
        AIC     = AIC(mod),
        BIC     = BIC(mod),
        BF_vs_reference =
          if (mdl == reference_model) 1
          else as.numeric(
            bayesfactor_models(mod, denominator = ref_mod)$log_BF[1]
          )
      )
      i <- i + 1
    }
  }

  bind_rows(model_metrics)
}