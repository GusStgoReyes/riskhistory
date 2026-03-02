import numpy as np

# =============================================================================
# LINK FUNCTIONS
# =============================================================================

def z_link_func(x, data=None):
    """
    Logistic link for starting point z, constraining output to (0, 1).
    Keeps z away from boundaries and makes it interpretable as a probability.
    """
    return 1 / (1 + np.exp(-x))


# =============================================================================
# DEFAULT MODEL PARAMETERS
# =============================================================================

DEFAULT_MODEL_PARAMS = {
    "model": "angle",
    "informative": False,
    "is_group_model": True,
    "group_only_regressors": False,
    "include": ["v", "a", "t", "z", "theta"],
}


# =============================================================================
# VARIABLE CODING CONVENTIONS
# =============================================================================
#
#   gain       : gain magnitude, mean-centered within each dataset.
#                  Intercept reflects drift at the average gamble, not at
#                  a zero-gain gamble that never occurs in the design.
#
#   loss       : loss magnitude, mean-centered within each dataset.
#                  Same rationale as gain.
#
#   choice_prev1 : previous choice, coded −1 (reject) or +1 (accept).
#                  Symmetric coding preserves the accept/reject symmetry
#                  of the design. The coefficient reflects the signed
#                  drift/z shift toward repeating the previous response.
#
#   pupil        : trial-level LSS pupil beta, z-scored within subject,
#                  then min-max scaled to [0, 1] within subject.
#                  0 = minimum observed arousal (weakest modulation),
#                  1 = maximum observed arousal (strongest modulation).
#                  [0, 1] scaling ensures pupil only ever adds to or
#                  scales a baseline effect — never reverses it, which
#                  would be theoretically incoherent.
#                  Used in z0 and v0 model families.
#
# =============================================================================
# MODEL REGISTRY
# =============================================================================
#
# NAMING CONVENTION:
#   HDDM_Collapse                         → Tier 0 baseline
#   HDDM_Collapse_Lag1-[mech]             → Tier 1/2 history models
#   HDDM_Collapse_Pupil_Lag1              → Tier 3 pupil baseline
#   HDDM_Collapse_Pupil_Lag1-[mech]       → Tier 4 single-mechanism pupil
#   HDDM_Collapse_Pupil_Lag1-[m1]_[m2].. → Tier 5 all-mechanism pupil
#
# MECHANISM CODES:
#   z0   → previous choice (and/or pupil) enters starting point z
#   v0   → previous choice (and/or pupil:prev_choice) enters drift v
#   stim → pupil rescales gain and loss weights in drift v
#
# MODEL TIERS:
#   Tier 0 : Baseline — no history, full sample
#   Tier 1 : Single-mechanism history — full sample
#   Tier 2 : All-mechanism history — full sample
#   Tier 3 : Pupil baseline — pupil subsample, history included,
#             no pupil regressors (BIC reference for Tiers 4–5)
#   Tier 4 : Single-mechanism pupil — pupil subsample
#   Tier 5 : All-mechanism pupil — pupil subsample
#
# group_only_nodes:
#   Regressors listed here are estimated at the group level only (single
#   posterior for the population effect). This applies to all history and
#   pupil effect-of-interest terms.
#   Per-subject posteriors for v_1, v_gain, v_loss, z_1, a, t, theta
#   are estimated automatically because group_only_regressors=False.
#
# =============================================================================

MODEL_REGISTRY = {

    # =========================================================================
    # TIER 0: BASELINE
    # =========================================================================
    # No history terms. Drift is a linear function of gain and loss only.
    # BIC reference for Tier 1 and Tier 2.
    # =========================================================================

    "HDDM_Collapse": {
        "description": (
            "Tier 0 baseline. Drift rate is a linear function of centered "
            "gain and loss. No history or pupil terms. Starting point z is "
            "free per subject but unregressed. BIC reference for Tiers 1–2."
        ),
        "tier": 0,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS,
    },

    # =========================================================================
    # TIER 1: SINGLE-MECHANISM CHOICE HISTORY
    # =========================================================================
    # Two competing hypotheses: previous choice acts on z OR on v.
    # Δk = 1 vs HDDM_Collapse for each model.
    # =========================================================================

    "HDDM_Collapse_Lag1-z0": {
        "description": (
            "Tier 1 — starting point hypothesis. Previous choice shifts z "
            "before evidence accumulation begins (pre-decisional bias toward "
            "repeating the prior response). Drift depends only on gain and "
            "loss. "
        ),
        "tier": 1,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss",  "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1",      "link_func": z_link_func},
        ],
        "group_only_nodes": ["z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-v0": {
        "description": (
            "Tier 1 — drift bias hypothesis. Previous choice adds a signed "
            "constant to the drift rate, biasing evidence accumulation toward "
            "repeating the prior response. Starting point z is free per "
            "subject but unregressed. "
        ),
        "tier": 1,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": ["v_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    # =========================================================================
    # TIER 2: ALL-MECHANISM CHOICE HISTORY
    # =========================================================================
    # Previous choice enters both z and v simultaneously.
    # Tests whether the two mechanisms account for independent variance.
    # Δk = 2 vs HDDM_Collapse.
    # =========================================================================

    "HDDM_Collapse_Lag1-z0_v0": {
        "description": (
            "Tier 2 — combined starting point + drift bias. Previous choice "
            "shifts z (pre-decisional) AND adds a signed drift constant "
            "(during accumulation). Tests whether both mechanisms co-occur "
            "or whether one fully accounts for the history effect. "
        ),
        "tier": 2,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_choice_prev1", "z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    # =========================================================================
    # TIER 3: PUPIL BASELINE
    # =========================================================================
    # Fit on the PUPIL SUBSAMPLE ONLY.
    # Retains both history terms (z and v) as the established mechanism.
    # No pupil regressors. BIC reference for all Tier 4 and Tier 5 models.
    # =========================================================================

    "HDDM_Collapse_Pupil_Lag1": {
        "description": (
            "Tier 3 — pupil subsample baseline. Previous choice enters both "
            "z and v (matching Tier 2 specification). No pupil regressors. "
            "Estimated on the pupil subsample only. "
            "BIC reference for all Tier 4 and Tier 5 pupil models."
        ),
        "tier": 3,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_choice_prev1", "z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    # =========================================================================
    # TIER 4: SINGLE-MECHANISM PUPIL
    # =========================================================================
    # Fit on the PUPIL SUBSAMPLE ONLY.
    # All models retain both history terms from Tier 3 so that pupil terms
    # explain variance beyond the established history effect.
    # Δk = 2 vs HDDM_Collapse_Pupil_Lag1 for each model.
    # =========================================================================

    "HDDM_Collapse_Pupil_Lag1-z0": {
        "description": (
            "Tier 4 — starting point pupil model. pupil and pupil:choice_prev1 "
            "enter z alongside the baseline history term. Tests whether arousal "
            "shifts the pre-decisional bias and whether that shift depends on "
            "the previous choice. Drift retains the history term but has no "
            "pupil terms. "
            "pupil: z-scored "
            "z_pupil:choice_prev1 = change in bias from min to max arousal. "
            "(z_pupil + z_pupil:choice_prev1)."
        ),
        "tier": 4,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + choice_prev1",
                "link_func": lambda x: x,
            },
            {
                "model": "z ~ 1 + choice_prev1 + pupil + pupil:choice_prev1",
                "link_func": z_link_func,
            },
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "z_choice_prev1",
            "z_pupil",
            "z_pupil:choice_prev1",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Pupil_Lag1-stim": {
        "description": (
            "Tier 4 — stimulus sensitivity pupil model. pupil_c rescales how "
            "gain and loss are weighted during evidence accumulation via "
            "pupil_c:gain and pupil_c:loss. History terms are retained in "
            "both v and z. Tests whether arousal modulates valuation sensitivity "
            "independent of the history bias. "
            "pupil ∈ [0, 1]: v_gain and v_loss are interpretable "
            "at average arousal (pupil_c = 0), reducing collinearity. "
            "Δk = 2 vs HDDM_Collapse_Pupil_Lag1 "
            "(v_pupil_c:gain + v_pupil_c:loss)."
        ),
        "tier": 4,
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain + loss + choice_prev1 "
                    "+ pupil:gain + pupil:loss"
                ),
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "z_choice_prev1",
            "v_pupil:gain",
            "v_pupil:loss",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Pupil_Lag1-stimEV": {
        "description": (
            "Tier 4 — expected-value sensitivity pupil model. pupil:ev rescales "
            "how the net expected value (ev) drives evidence accumulation. History "
            "terms are retained in both v and z. Tests whether arousal modulates "
            "overall valuation sensitivity via a single EV term rather than "
            "separate gain/loss terms. "
            "Δk = 1 vs HDDM_Collapse_Pupil_Lag1 "
            "(v_pupil:ev)."
        ),
        "tier": 4,
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain + loss + choice_prev1 "
                    "+ pupil:ev"
                ),
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "z_choice_prev1",
            "v_pupil:ev",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Pupil_Lag1-v0": {
        "description": (
            "Tier 4 — drift bias pupil model. pupil and pupil:choice_prev1 "
            "enter the drift rate alongside the baseline history term. Tests "
            "whether arousal modulates the drift-level repetition bias during "
            "evidence accumulation. Starting point z retains only the history "
            "term. "
            "pupil ∈ [0, 1]: v_choice_prev1 = drift history bias at minimum "
            "arousal; v_pupil:choice_prev1 = change in bias from min to max "
            "arousal. "
            "Δk = 2 vs HDDM_Collapse_Pupil_Lag1 "
            "(v_pupil + v_pupil:choice_prev1)."
        ),
        "tier": 4,
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain + loss + choice_prev1 "
                    "+ pupil + pupil:choice_prev1"
                ),
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "z_choice_prev1",
            "v_pupil",
            "v_pupil:choice_prev1",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

    # =========================================================================
    # TIER 5: ALL-MECHANISM PUPIL
    # =========================================================================
    # All three pupil mechanisms active simultaneously.
    # Fit on the PUPIL SUBSAMPLE ONLY.
    # Expected to be over-parametrized relative to Tier 4 single-mechanism
    # models under BIC. Included to verify that no important variance is
    # missed by the simpler models.
    #
    # Requires TWO pupil columns in the dataframe:
    #   pupil   ∈ [0, 1]       for z0 and v0 terms
    #   pupil_c ∈ [−0.5,+0.5]  for stim terms
    # Both are derived from the same z-scored LSS beta; only the shift differs.
    # =========================================================================

    "HDDM_Collapse_Pupil_Lag1-z0_v0_stim": {
        "description": (
            "Tier 5 — full pupil model. All three pupil mechanisms active: "
            "(1) z: pupil and pupil:choice_prev1 shift the pre-decisional bias "
            "(pupil ∈ [0,1]); "
            "(2) v: pupil and pupil:choice_prev1 modulate the drift-level "
            "repetition bias (pupil ∈ [0,1]); "
            "(3) v: pupil:gain and pupil:loss rescale valuation "
            "sensitivity (pupil ∈ [0,1]). "
            "Δk = 6 vs HDDM_Collapse_Pupil_Lag1 "
            "(z_pupil, z_pupil:prev, v_pupil, v_pupil:prev, "
            "v_pupil:gain, v_pupil:loss)."
        ),
        "tier": 5,
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain + loss + choice_prev1 "
                    "+ pupil + pupil:choice_prev1 "
                    "+ pupil:gain + pupil:loss"
                ),
                "link_func": lambda x: x,
            },
            {
                "model": "z ~ 1 + choice_prev1 + pupil + pupil:choice_prev1",
                "link_func": z_link_func,
            },
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "z_choice_prev1",
            "z_pupil",
            "z_pupil:choice_prev1",
            "v_pupil",
            "v_pupil:choice_prev1",
            "v_pupil:gain",
            "v_pupil:loss",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Pupil_Lag1-z0_v0_stimEV": {
        "description": (
            "Tier 5 — full pupil model with EV sensitivity. All three pupil "
            "mechanisms active: "
            "(1) z: pupil and pupil:choice_prev1 shift the pre-decisional bias "
            "(pupil ∈ [0,1]); "
            "(2) v: pupil and pupil:choice_prev1 modulate the drift-level "
            "repetition bias (pupil ∈ [0,1]); "
            "(3) v: pupil:ev rescales overall valuation sensitivity. "
            "Δk = 5 vs HDDM_Collapse_Pupil_Lag1 "
            "(z_pupil, z_pupil:prev, v_pupil, v_pupil:prev, v_pupil:ev)."
        ),
        "tier": 5,
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain + loss + choice_prev1 "
                    "+ pupil + pupil:choice_prev1 "
                    "+ pupil:ev"
                ),
                "link_func": lambda x: x,
            },
            {
                "model": "z ~ 1 + choice_prev1 + pupil + pupil:choice_prev1",
                "link_func": z_link_func,
            },
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "z_choice_prev1",
            "z_pupil",
            "z_pupil:choice_prev1",
            "v_pupil",
            "v_pupil:choice_prev1",
            "v_pupil:ev",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

}


# =============================================================================
# MODEL COMPARISON ORDER
# =============================================================================
#
# STEP 1 — Tier 0 vs Tier 1 (full sample)
#   Establishes whether previous choice improves fit and which single
#   mechanism (z or v) carries the history effect.
#   BIC reference: HDDM_Collapse.
#
# STEP 2 — Tier 2 (full sample)
#   Tests whether both mechanisms together outperform either alone.
#   Run only if Tier 1 is not decisive (ΔBIC < 2 between z0 and v0).
#   BIC reference: HDDM_Collapse.
#
# STEP 3 — Tier 3 vs Tier 4 (pupil subsample)
#   Single-mechanism pupil models compared against HDDM_Collapse_Pupil_Lag1.
#   BIC reference: HDDM_Collapse_Pupil_Lag1.
#
# STEP 4 — Tier 5 (pupil subsample)
#   Full model checked last; expected to lose under BIC.
#   BIC reference: HDDM_Collapse_Pupil_Lag1.
#
# =============================================================================

MODEL_COMPARISON_ORDER = [
    # ── Tier 0 ───────────────────────────────────────────────────────────────
    "HDDM_Collapse",
    # ── Tier 1 ───────────────────────────────────────────────────────────────
    "HDDM_Collapse_Lag1-z0",
    "HDDM_Collapse_Lag1-v0",
    # ── Tier 2 ───────────────────────────────────────────────────────────────
    "HDDM_Collapse_Lag1-z0_v0",
    # ── Tier 3: pupil subsample BIC reference ────────────────────────────────
    "HDDM_Collapse_Pupil_Lag1",
    # ── Tier 4: single-mechanism pupil ───────────────────────────────────────
    "HDDM_Collapse_Pupil_Lag1-z0",
    "HDDM_Collapse_Pupil_Lag1-stim",
    "HDDM_Collapse_Pupil_Lag1-stimEV",
    "HDDM_Collapse_Pupil_Lag1-v0",
    # ── Tier 5: all-mechanism pupil ──────────────────────────────────────────
    "HDDM_Collapse_Pupil_Lag1-z0_v0_stim",
    "HDDM_Collapse_Pupil_Lag1-z0_v0_stimEV",
]


# =============================================================================
# PARAMETER COUNT TABLE (for BIC penalty accounting)
# =============================================================================
#
# Δk = group-level parameters added vs. the appropriate tier reference.
#
# Tier 0–2 reference : HDDM_Collapse            (Δk = 0)
# Tier 3–5 reference : HDDM_Collapse_Pupil_Lag1 (Δk = 0, pupil subsample)
#
# group_only_nodes contribute exactly 1 param each (group mean only).
# Per-subject v_1, v_gain, v_loss, z_1, a, t, theta posteriors are
# free in all models and are not counted here (they do not differ across
# models within a tier).
#
# BIC penalty per param = ln(N_trials).
# For N ≈ 10k–50k trials: ≈ 9–11 log-likelihood units per parameter.
#
PARAM_DELTA_FROM_BASELINE = {
    # Tier 0
    "HDDM_Collapse":                         0,
    # Tier 1 — Δk vs HDDM_Collapse
    "HDDM_Collapse_Lag1-z0":                1,   # z_choice_prev1
    "HDDM_Collapse_Lag1-v0":                1,   # v_choice_prev1
    # Tier 2 — Δk vs HDDM_Collapse
    "HDDM_Collapse_Lag1-z0_v0":             2,   # z_choice_prev1 + v_choice_prev1
    # Tier 3 — reference for pupil subsample (Δk = 0 by definition)
    "HDDM_Collapse_Pupil_Lag1":             0,
    # Tier 4 — Δk vs HDDM_Collapse_Pupil_Lag1
    "HDDM_Collapse_Pupil_Lag1-z0":          2,   # z_pupil + z_pupil:choice_prev1
    "HDDM_Collapse_Pupil_Lag1-stim":        2,   # v_pupil:gain + v_pupil:loss
    "HDDM_Collapse_Pupil_Lag1-stimEV":      1,   # v_pupil:ev
    "HDDM_Collapse_Pupil_Lag1-v0":          2,   # v_pupil + v_pupil:choice_prev1
    # Tier 5 — Δk vs HDDM_Collapse_Pupil_Lag1
    "HDDM_Collapse_Pupil_Lag1-z0_v0_stim":    6,   # z0(2) + v0(2) + stim(2)
    "HDDM_Collapse_Pupil_Lag1-z0_v0_stimEV":  5,   # z0(2) + v0(2) + stimEV(1)
}


# =============================================================================
# MODEL ACCESSOR FUNCTIONS
# =============================================================================

def get_available_models() -> list:
    """Return list of all registered model IDs."""
    return list(MODEL_REGISTRY.keys())


def get_model_tier(tier: int) -> list:
    """Return all model IDs belonging to a given tier."""
    return [k for k, v in MODEL_REGISTRY.items() if v.get("tier") == tier]


def get_model_description(model_id: str) -> str:
    """Return the description string for a model."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    return MODEL_REGISTRY[model_id]["description"]


def get_model_config(model_id: str) -> dict:
    """
    Return the HDDM config dict for a model, stripping metadata keys
    (description, tier) that are not passed to HDDMnnRegressor.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    config = MODEL_REGISTRY[model_id].copy()
    config.pop("description", None)
    config.pop("tier", None)
    return config


def get_param_delta(model_id: str) -> int:
    """Return the number of extra group-level parameters vs. the tier reference."""
    if model_id not in PARAM_DELTA_FROM_BASELINE:
        raise ValueError(f"No param delta registered for: {model_id}")
    return PARAM_DELTA_FROM_BASELINE[model_id]