# Extracts per-subject total slopes (random deviation + fixed effect) for
# accept and switch from a fitted model list. Returns one row per subject
# per dataset. Columns are absent if the model has no random slope for that
# term (e.g., after simplifying the random effects structure).
extract_blups <- function(model_list, datasets, model_name) {
  results <- list()

  for (ds in datasets) {
    m  <- model_list[[ds]][[model_name]]
    re <- as.data.frame(ranef(m)$sub)
    fe <- fixef(m)

    blup_df <- data.frame(sub = rownames(re), dataset = ds,
                          stringsAsFactors = FALSE)

    if ("accept" %in% colnames(re)) {
      blup_df$slope_accept <- re[["accept"]] + fe["accept"]
    }

    if ("switch" %in% colnames(re)) {
      blup_df$slope_switch <- re[["switch"]] + fe["switch"]
    }

    rownames(blup_df) <- NULL
    results[[ds]] <- blup_df
  }

  do.call(rbind, results)
}
