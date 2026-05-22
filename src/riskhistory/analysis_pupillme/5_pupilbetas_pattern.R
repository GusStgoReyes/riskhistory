# =============================================================================
# Choice History Under Risk: Pupil Betas — Pattern Model (M0–M3)
# =============================================================================

library(bayestestR)

acceptglme_path <- file.path(repo_path, "src", "riskhistory", "analysis_acceptglme")
source(file.path(acceptglme_path, "utils_modelcomparison.R"))
source(file.path(acceptglme_path, "utils_savefixedeffects.R"))

# =============================================================================
# Add pattern variable
# =============================================================================
data_betas_pattern <- add_pattern(data_betas_clean)

message("Pattern cell counts (all datasets):")
print(table(data_betas_pattern$pattern, useNA = "ifany"))

sparse <- data_betas_pattern %>%
  filter(!is.na(pattern)) %>%
  group_by(sub, dataset, pattern) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n < 10)

if (nrow(sparse) > 0) {
  message("WARNING: subjects with <10 trials in a pattern cell:")
  print(as.data.frame(sparse))
}

# =============================================================================
# Fit M0–M3 per dataset (REML=FALSE for LRT)
# =============================================================================
pupilpattern_models_list <- setNames(lapply(datasets, function(x) list()), datasets)
lrt_list <- list()

for (ds in datasets) {
  message(sprintf("\nFitting pattern models for dataset: %s", ds))
  df_sub <- data_betas_pattern %>%
    filter(dataset == ds, !is.na(pattern)) %>%
    tidyr::drop_na(pupil, logRT_z, gain_z, loss_z, entropy_c)

  M0 <- lmer(pupil ~ logRT_z +
               (1 + logRT_z | sub),
             data = df_sub, REML = FALSE, control = LMER_CONTROL)

  M1 <- lmer(pupil ~ logRT_z + gain_z + loss_z +
               (1 + logRT_z | sub),
             data = df_sub, REML = FALSE, control = LMER_CONTROL)

  M2 <- lmer(pupil ~ logRT_z + gain_z + loss_z + entropy_c +
               (1 + logRT_z | sub),
             data = df_sub, REML = FALSE, control = LMER_CONTROL)

  M3 <- lmer(pupil ~ logRT_z + gain_z + loss_z + entropy_c + pattern +
               (1 + logRT_z | sub),
             data = df_sub, REML = FALSE, control = LMER_CONTROL)
  print(summary(M3))

  diag <- lapply(list(M0 = M0, M1 = M1, M2 = M2, M3 = M3),
                 check_model_diagnostics)
  for (nm in names(diag)) {
    message(sprintf("  %s — converged: %s, singular: %s",
                    nm, diag[[nm]]$converged, diag[[nm]]$singular))
  }

  pupilpattern_models_list[[ds]] <- list(M0 = M0, M1 = M1, M2 = M2, M3 = M3)

  # lrt <- anova(M0, M1, M2, M3)
  # lrt_df <- as.data.frame(lrt)
  # lrt_df$step <- rownames(lrt_df)
  # lrt_df$dataset <- ds
  # lrt_list[[ds]] <- lrt_df
}

# =============================================================================
# Save convergence report
# =============================================================================
convergence_rows <- lapply(datasets, function(ds) {
  lapply(names(pupilpattern_models_list[[ds]]), function(nm) {
    d <- check_model_diagnostics(pupilpattern_models_list[[ds]][[nm]])
    data.frame(dataset = ds, model = nm,
               converged = d$converged, singular = d$singular,
               stringsAsFactors = FALSE)
  })
})
convergence_pattern <- do.call(rbind, unlist(convergence_rows, recursive = FALSE))

write.csv(convergence_pattern,
          file.path(analysis_results_path, "5_pupilbetas_pattern_convergence.csv"),
          row.names = FALSE)
message("Saved: 5_pupilbetas_pattern_convergence.csv")

# =============================================================================
# LRT (nested sequence)
# =============================================================================
lrt_all <- do.call(rbind, lrt_list)
rownames(lrt_all) <- NULL

write.csv(lrt_all,
          file.path(analysis_results_path, "5_pupilbetas_pattern_lrt.csv"),
          row.names = FALSE)
message("Saved: 5_pupilbetas_pattern_lrt.csv")

# =============================================================================
# AIC / BIC / BF (reference = M0)
# =============================================================================
model_comparison <- compare_models(
  model_list      = pupilpattern_models_list,
  datasets        = datasets,
  models          = c("M0", "M1", "M2", "M3"),
  reference_model = "M0"
)

write.csv(model_comparison,
          file.path(analysis_results_path,
                    "5_pupilbetas_pattern_model_comparison.csv"),
          row.names = FALSE)
message("Saved: 5_pupilbetas_pattern_model_comparison.csv")

# =============================================================================
# Refit M3 with REML=TRUE for coefficient reporting
# =============================================================================
pupilpattern_m3_reml <- setNames(lapply(datasets, function(x) list()), datasets)

for (ds in datasets) {
  df_sub <- data_betas_pattern %>%
    filter(dataset == ds, !is.na(pattern)) %>%
    tidyr::drop_na(pupil, logRT_z, gain_z, loss_z, entropy_c)

  m3_reml <- lmer(pupil ~ logRT_z + gain_z + loss_z + entropy_c + pattern +
                    (1 + logRT_z | sub),
                  data = df_sub, REML = TRUE, control = LMER_CONTROL)

  pupilpattern_m3_reml[[ds]] <- list(M3_reml = m3_reml)
}

# =============================================================================
# Fixed effects (REML=TRUE)
# =============================================================================
fixed_effects <- extract_fixed_effects(
  model_list = pupilpattern_m3_reml,
  datasets   = datasets,
  models     = "M3_reml",
  p_method   = p_method
)

write.csv(fixed_effects,
          file.path(analysis_results_path,
                    "5_pupilbetas_pattern_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 5_pupilbetas_pattern_fixed_effects.csv")

# =============================================================================
# Per-subject random slopes for patternstreak_break (for individual differences)
# =============================================================================
re_streak_break <- lapply(datasets, function(ds) {
  m3 <- pupilpattern_m3_reml[[ds]]$M3_reml
  re <- ranef(m3)$sub
  if (!"patternstreak_break" %in% colnames(re)) {
    message(sprintf("  %s: no random slope for patternstreak_break", ds))
    return(NULL)
  }
  fe <- fixef(m3)["patternstreak_break"]
  re %>%
    as.data.frame() %>%
    tibble::rownames_to_column("sub") %>%
    transmute(sub,
              slope_streak_break = patternstreak_break + fe,
              dataset = ds)
}) %>%
  Filter(Negate(is.null), .) %>%
  bind_rows()

if (nrow(re_streak_break) > 0) {
  write.csv(re_streak_break,
            file.path(analysis_results_path,
                      "5_pupilbetas_pattern_re_streak_break.csv"),
            row.names = FALSE)
  message("Saved: 5_pupilbetas_pattern_re_streak_break.csv")
}

message("\nPattern model analysis complete.")
