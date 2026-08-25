# =============================================================================
# Pupil-Behavior — Replicate RT Model
# =============================================================================
# log(rt) ~ sv_abs + transition + (1 + sv_abs | sub)

formula_replicate_rt <- log(rt) ~ sv_abs + transition + (1 + sv_abs | sub)

# Same two transition contrasts used in 4_pupilrtmodel.R / analysis_pupiltimeseries.
TRANSITION_PAIRS <- list(
  c("accept_to_reject", "reject_to_reject"),
  c("reject_to_accept", "accept_to_accept")
)

# avg_comparisons(variables = list(transition = "pairwise")) always labels
# each pair as "later factor level - earlier factor level" (by transition's
# own level order: accept_to_accept, accept_to_reject, reject_to_accept,
# reject_to_reject) -- not necessarily pair[1] - pair[2] as written above.
# For each (A, B) in contrast_pairs, find whichever of "A - B" / "B - A" is
# present and normalize it to "A - B", taking the reciprocal of the ratio
# when the reversed label was the one found (p.value is direction-invariant
# for a ratio-vs-1 test either way).
normalize_pairwise_ratio_contrasts <- function(comp_df, contrast_pairs) {
  matched_rows <- lapply(contrast_pairs, function(pair) {
    label_fwd <- paste(pair[1], "-", pair[2])
    row_fwd <- comp_df[comp_df$contrast == label_fwd, ]
    if (nrow(row_fwd) > 0) {
      row_fwd$contrast <- label_fwd
      return(row_fwd)
    }

    label_rev <- paste(pair[2], "-", pair[1])
    row_rev <- comp_df[comp_df$contrast == label_rev, ]
    row_rev$contrast <- label_fwd
    row_rev$estimate <- 1 / row_rev$estimate
    row_rev
  })
  do.call(rbind, matched_rows)
}

replicatertmodel_list <- list()
replicatertmodel_contrasts_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting replicate RT model for dataset: %s", ds))
  df <- data_pupilbehavior %>% filter(dataset == ds)

  m <- lmer(formula_replicate_rt, data = df, control = LMER_CONTROL)
  diag <- check_model_diagnostics(m)
  message(sprintf("  converged: %s, singular: %s", diag$converged, diag$singular))

  replicatertmodel_list[[ds]] <- list(replicate_rt = m)

  # A-R vs R-R and R-A vs A-A contrasts
  message(sprintf("  Computing avg_comparisons for dataset: %s", ds))
  comp <- avg_comparisons(m, variables = list(transition = "pairwise"),
                          transform = "exp")
  comp_df <- normalize_pairwise_ratio_contrasts(as.data.frame(comp), TRANSITION_PAIRS)

  replicatertmodel_contrasts_list[[ds]] <- data.frame(
    dataset  = ds,
    contrast = comp_df$contrast,
    estimate = comp_df$estimate,
    p.value  = comp_df$p.value,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Convergence report
# =============================================================================
convergence_df <- check_convergence_all_models(
  replicatertmodel_list, datasets, c("replicate_rt")
)

write.csv(convergence_df,
          file.path(analysis_results_path,
                    "1_replicate_rtmodel_convergence.csv"),
          row.names = FALSE)
message("Saved: 1_replicate_rtmodel_convergence.csv")

# =============================================================================
# A-R vs R-R / R-A vs A-A RT contrasts (ratio scale, from transform = "exp")
# =============================================================================
replicatertmodel_contrasts_df <- bind_rows(replicatertmodel_contrasts_list)
replicatertmodel_contrasts_df$p_adj_bh <- ave(
  replicatertmodel_contrasts_df$p.value, replicatertmodel_contrasts_df$dataset,
  FUN = function(p) p.adjust(p, method = p_method)
)

write.csv(replicatertmodel_contrasts_df,
          file.path(analysis_results_path,
                    "1_replicate_rtmodel_contrasts.csv"),
          row.names = FALSE)
message("Saved: 1_replicate_rtmodel_contrasts.csv")

message("\nReplicate RT model analysis complete.")
