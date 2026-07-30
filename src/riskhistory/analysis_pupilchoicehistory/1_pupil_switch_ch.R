# =============================================================================
# Pupil Switch ~ Choice-History DDM Biases
# =============================================================================
# Per dataset: is each subject's "pupil switch" signature explained by their
# individual choice-history DDM biases?
#
#   pupil_switch ~ ch_stp + ch_drift_bias + ch_valuation_bias
#
# pupil_switch = mean(AR, RA) - mean(AA, RR): the average pupil beta on
# switch trials (accept_to_reject, reject_to_accept) minus the average on
# repeat trials (accept_to_accept, reject_to_reject).
#
# ch_stp / ch_drift_bias / ch_valuation_bias are per-subject analogs of
# figure3.ipynb's group-posterior Pre-Valuation/Drift/Valuation Bias metrics
# (from HDDM_Base_Lag1-full's z_Intercept/z_choice_prev1, v_Intercept/
# v_choice_prev1, v_gain_centered(:choice_prev1)/v_loss_centered(:choice_prev1)
# terms). Each is the accept-minus-reject difference of that subject's own
# point estimate (instead of two separate group-posterior distributions):
#   ch_stp             = starting_point_accept - starting_point_reject
#   ch_drift_bias      = drift_bias_accept - drift_bias_reject
#   ch_valuation_bias  = lambda_accept - lambda_reject
# =============================================================================

z_link_func <- function(x) 1 / (1 + exp(-x))

FULL_DDM_MODEL <- "HDDM_Base_Lag1-full"
hddm_params_path <- file.path(results_path, "analysis_ddm", "hddm_params")

# =============================================================================
# Per-subject choice-history DDM bias metrics
# =============================================================================
message("Loading per-subject HDDM_Base_Lag1-full parameters...")
ch_bias_df <- bind_rows(lapply(datasets, function(ds) {
  read.csv(file.path(hddm_params_path,
                     sprintf("%s_%s_subj_params.csv", FULL_DDM_MODEL, ds)))
})) %>%
  select(sub, dataset, param_name, param_value) %>%
  pivot_wider(names_from = param_name, values_from = param_value) %>%
  mutate(
    starting_point_accept = z_link_func(z_Intercept + z_choice_prev1),
    starting_point_reject = z_link_func(z_Intercept - z_choice_prev1),
    ch_stp = starting_point_accept - starting_point_reject,

    drift_bias_accept = v_Intercept + v_choice_prev1,
    drift_bias_reject = v_Intercept - v_choice_prev1,
    ch_drift_bias = drift_bias_accept - drift_bias_reject,

    w_gain_accept = v_gain_centered + `v_gain_centered:choice_prev1`,
    w_gain_reject = v_gain_centered - `v_gain_centered:choice_prev1`,
    w_loss_accept = v_loss_centered + `v_loss_centered:choice_prev1`,
    w_loss_reject = v_loss_centered - `v_loss_centered:choice_prev1`,
    lambda_accept = abs(w_loss_accept / w_gain_accept),
    lambda_reject = abs(w_loss_reject / w_gain_reject),
    ch_valuation_bias = lambda_accept - lambda_reject
  ) %>%
  select(sub, dataset, ch_stp, ch_drift_bias, ch_valuation_bias)

# =============================================================================
# Per-subject pupil switch score
# =============================================================================
message("Building per-subject pupil beta by transition...")
pupil_transition_df <- build_pupil_transition_df(data_betas_clean)

pupil_switch_df <- pupil_transition_df %>%
  pivot_wider(names_from = transition, values_from = pupil_beta) %>%
  mutate(pupil_switch = (AR + RA) / 2 - (AA + RR) / 2) %>%
  select(sub, dataset, pupil_switch)

model_data <- pupil_switch_df %>%
  inner_join(ch_bias_df, by = c("sub", "dataset")) %>%
  filter(if_all(c(pupil_switch, ch_stp, ch_drift_bias, ch_valuation_bias), ~ !is.na(.)))

# =============================================================================
# Fit one model per dataset
# =============================================================================
switch_models_list <- list()
fixed_effects_rows <- list()

for (ds in datasets) {
  message(sprintf("\nFitting pupil_switch ~ ch_stp + ch_drift_bias + ch_valuation_bias for dataset: %s", ds))

  df_ds <- model_data %>% filter(dataset == ds)
  m <- lm(pupil_switch ~ ch_stp + ch_drift_bias + ch_valuation_bias, data = df_ds)
  switch_models_list[[ds]] <- m

  message(sprintf("  n_subs=%d", nrow(df_ds)))

  fe <- broom::tidy(m, conf.int = TRUE) %>%
    mutate(model = "pupil_switch_ch", dataset = ds, n_subs = nrow(df_ds))
  fixed_effects_rows[[length(fixed_effects_rows) + 1]] <- fe
}

fixed_effects_df <- bind_rows(fixed_effects_rows)

# BH-adjust each predictor's p-value across the 3 datasets
fixed_effects_df$p_adj_bh <- ave(
  fixed_effects_df$p.value,
  fixed_effects_df$term,
  FUN = function(p) p.adjust(p, method = p_method)
)

write.csv(fixed_effects_df,
          file.path(analysis_results_path, "1_pupil_switch_ch_fixed_effects.csv"),
          row.names = FALSE)
message("Saved: 1_pupil_switch_ch_fixed_effects.csv")

message("\nPupil switch ~ choice-history bias models complete.")
