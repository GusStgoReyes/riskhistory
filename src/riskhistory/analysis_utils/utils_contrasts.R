# =============================================================================
# Normalize marginaleffects pairwise contrast direction/scale
# =============================================================================
# avg_comparisons(model, variables = list(<factor> = "pairwise")) returns all
# pairwise contrasts between factor levels, but its contrast labels follow
# marginaleffects' own internal level ordering, not necessarily the order the
# caller wants. For each (A, B) pair in contrast_pairs, find whichever of
# "A - B" / "B - A" is present and normalize it to "A - B", inverting the
# reversed row's estimate accordingly:
#   invert = "sign"       -- negate estimate/statistic (additive/log-odds scale)
#   invert = "reciprocal" -- take 1/estimate (ratio scale, e.g. transform = "exp")
# p.value is unaffected either way (two-tailed / ratio-vs-1 tests are
# direction-invariant). When invert = "sign", the return value is narrowed to
# contrast/estimate/SE/z.ratio/p.value; when invert = "reciprocal", the full
# comp_df (with estimate/conf.low/conf.high modified, if present) is returned
# unchanged otherwise -- matching the two call sites' pre-existing conventions
# (analysis_pupiltimeseries's timepoint models and analysis_pupilbehavior's RT
# model, respectively).
normalize_pairwise_contrasts <- function(comp_df, contrast_pairs,
                                         invert = c("sign", "reciprocal")) {
  invert <- match.arg(invert)

  matched_rows <- lapply(contrast_pairs, function(pair) {
    label_fwd <- paste(pair[1], "-", pair[2])
    row_fwd <- comp_df[comp_df$contrast == label_fwd, ]
    if (nrow(row_fwd) > 0) {
      row_fwd$contrast <- label_fwd
      return(row_fwd)
    }

    label_rev <- paste(pair[2], "-", pair[1])
    row_rev <- comp_df[comp_df$contrast == label_rev, ]
    if (nrow(row_rev) == 0) return(NULL)

    row_rev$contrast <- label_fwd
    if (invert == "sign") {
      row_rev$estimate  <- -row_rev$estimate
      row_rev$statistic <- -row_rev$statistic
      if ("conf.low" %in% names(row_rev)) {
        old_low  <- row_rev$conf.low
        old_high <- row_rev$conf.high
        row_rev$conf.low  <- -old_high
        row_rev$conf.high <- -old_low
      }
    } else {
      if ("conf.low" %in% names(row_rev)) {
        old_low  <- row_rev$conf.low
        old_high <- row_rev$conf.high
        row_rev$conf.low  <- 1 / old_high
        row_rev$conf.high <- 1 / old_low
      }
      row_rev$estimate <- 1 / row_rev$estimate
    }
    row_rev
  })
  matched_rows <- matched_rows[!vapply(matched_rows, is.null, logical(1))]
  if (length(matched_rows) == 0) return(NULL)

  comp_df <- do.call(rbind, matched_rows)

  if (invert == "sign") {
    data.frame(
      contrast = comp_df$contrast,
      estimate = comp_df$estimate,
      SE       = comp_df$std.error,
      z.ratio  = comp_df$statistic,
      p.value  = comp_df$p.value,
      stringsAsFactors = FALSE
    )
  } else {
    comp_df
  }
}
