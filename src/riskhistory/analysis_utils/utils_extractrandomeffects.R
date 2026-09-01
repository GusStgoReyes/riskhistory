# =============================================================================
# Extract per-subject BLUPs (fixed + random effect) from an lmer/glmer model
# =============================================================================
extract_random_effects <- function(model, ds, coef_map) {
  # coef_map: named vector mapping model term names (exactly as they appear
  # in fixef()/ranef()
  fixef_vals <- fixef(model)
  ranef_vals <- ranef(model)$sub %>%
    rownames_to_column("sub")
  ranef_vals$sub <- as.integer(ranef_vals$sub)
  ranef_vals$dataset <- ds

  out <- ranef_vals %>% select(sub, dataset)
  for (term in names(coef_map)) {
    out[[paste0("beta_", coef_map[[term]])]] <- fixef_vals[[term]] + ranef_vals[[term]]
  }
  out
}
