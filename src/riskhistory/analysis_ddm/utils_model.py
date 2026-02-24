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
# MODEL REGISTRY
# =============================================================================
#
# NAMING CONVENTION:
#   HDDM_Collapse                        → baseline (no history)
#   HDDM_Collapse_Lag1-[mechanism]       → previous choice only
#   HDDM_Collapse_Lag1-[mech]_Pupil      → best prev-choice model + pupil
#
# MECHANISM CODES:
#   x0   → starting point z
#   c0   → drift bias (additive v term)
#   stim → stimulus coding (gain:prev, loss:prev interactions)
#   ndt  → non-decision time t
#
# MODEL TIERS:
#   Tier 0: Baseline
#   Tier 1: Previous choice mechanisms (model comparison to find winning mechanism)
#   Tier 2: Combined previous choice (z + best drift model)
#   Tier 3: Pupil models (build on winning Tier 1/2 model — z is expected winner)
#
# group_only_nodes:
#   Any regressor you want estimated at the group level only (no per-subject
#   posterior). Use for the effect of interest (previous choice, pupil) so that
#   a single coefficient captures the population-level mechanism.
#   Per-subject random effects for intercept and loss are handled by
#   group_only_regressors=False (default), meaning v_1, v_gain, v_loss, z_1
#   etc. get subject-level posteriors automatically.
#
# =============================================================================

MODEL_REGISTRY = {

    # -------------------------------------------------------------------------
    # TIER 0: BASELINE
    # -------------------------------------------------------------------------

    "HDDM_Collapse": {
        "description": (
            "Baseline model. Drift rate is a linear function of gain and loss "
            "magnitude. No history terms. All other DDM parameters (a, t, z, "
            "theta) are free per subject."
        ),
        "tier": 0,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS,
    },

    # -------------------------------------------------------------------------
    # TIER 1: PREVIOUS CHOICE — SINGLE MECHANISM
    # Four competing hypotheses about where previous choice acts.
    # -------------------------------------------------------------------------

    "HDDM_Collapse_Lag1-x0": {
        "description": (
            "Starting point hypothesis. Previous choice shifts z before "
            "evidence accumulation begins, reflecting a pre-decisional bias "
            "toward repeating the prior response. Drift is unchanged."
        ),
        "tier": 1,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss",       "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1",       "link_func": z_link_func},
        ],
        "group_only_nodes": ["z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-c0": {
        "description": (
            "Drift bias hypothesis. Previous choice adds a signed constant to "
            "the drift rate, biasing evidence accumulation toward repeating "
            "the prior response. Starting point is free but unregressed."
        ),
        "tier": 1,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss + choice_prev1", "link_func": lambda x: x},
        ],
        "group_only_nodes": ["v_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-stim": {
        "description": (
            "Stimulus sensitivity hypothesis. Previous choice modulates how "
            "gain and loss magnitudes are weighted during accumulation via "
            "gain:choice_prev1 and loss:choice_prev1 interactions. Tests "
            "whether history asymmetrically rescales gain vs. loss processing "
            "rather than adding a uniform drift bias."
        ),
        "tier": 1,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + gain:choice_prev1 + loss:choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": ["v_gain:choice_prev1", "v_loss:choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-ndt": {
        "description": (
            "Non-decision time hypothesis (control model). Previous choice "
            "affects t (motor preparation / encoding time) rather than the "
            "accumulation process itself. Uses 'repeat' (binary: did the "
            "participant repeat their previous choice?) as the predictor "
            "because NDT should be faster for prepared/repeated responses. "
            "If this model wins it would suggest a purely motor account."
        ),
        "tier": 1,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
            {"model": "t ~ 1 + repeat",      "link_func": lambda x: x},
        ],
        "group_only_nodes": ["t_repeat"],
        **DEFAULT_MODEL_PARAMS,
    },

    # -------------------------------------------------------------------------
    # TIER 2: COMBINED PREVIOUS CHOICE
    # Tests whether z and drift bias account for independent variance.
    # Only z+c0 combination is theoretically motivated here — z+stim is also
    # included to check if stimulus sensitivity adds on top of z shift.
    # -------------------------------------------------------------------------

    "HDDM_Collapse_Lag1-x0_c0": {
        "description": (
            "Combined starting point + drift bias. Previous choice shifts z "
            "(pre-decisional) AND adds a signed drift constant (during "
            "accumulation). Tests whether both mechanisms co-occur or whether "
            "one fully explains the other."
        ),
        "tier": 2,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss + choice_prev1", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1",               "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_choice_prev1", "z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-x0_stim": {
        "description": (
            "Combined starting point + stimulus sensitivity. Previous choice "
            "shifts z AND modulates gain/loss weights in the drift. Asks "
            "whether pre-decisional bias and evidence-weighting bias are "
            "separable or redundant."
        ),
        "tier": 2,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + gain:choice_prev1 + loss:choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [
            "v_gain:choice_prev1",
            "v_loss:choice_prev1",
            "z_choice_prev1",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

    # -------------------------------------------------------------------------
    # TIER 3: PUPIL MODELS
    #
    # Logic: previous choice mechanism is settled by Tier 1/2 comparison
    # (z is expected winner based on your existing results). Pupil enters
    # exclusively in the drift as pupil:choice_prev1, motivated by:
    #   (a) pupil is measured post-stimulus (during accumulation, not before)
    #   (b) logistic regression shows pupil:choice_prev1 significant, not pupil alone
    #
    # All pupil models keep z ~ 1 + choice_prev1 for the pre-decisional bias.
    # The pupil term tests whether arousal during accumulation modulates the
    # STRENGTH of the history-driven drift — i.e., whether high arousal
    # amplifies or attenuates the repetition bias in evidence accumulation.
    #
    # Three pupil models of increasing complexity:
    #   Pupil-v    : overall drift modulation by pupil:prev_choice
    #   Pupil-v0   : pupil as additive drift term WITHOUT interaction (null for pupil)
    #   Pupil-v_x0 : full model — z shift + pupil:prev_choice in drift
    # -------------------------------------------------------------------------

    "HDDM_Collapse_Lag1-x0_Pupil-v0": {
        "description": (
            "Pupil null model. Previous choice shifts z. Pupil enters drift "
            "as a main effect only (no interaction with previous choice). "
            "This is a null/control model for pupil — if pupil:prev_choice "
            "models beat this, it confirms the interaction is necessary. "
            "Expected to lose based on logistic regression showing pupil "
            "alone is non-significant."
        ),
        "tier": 3,
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss + pupil",  "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1",          "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_pupil", "z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-x0_Pupil-v": {
        "description": (
            "Primary pupil model. Previous choice shifts z (pre-decisional). "
            "Pupil modulates drift via pupil:choice_prev1 interaction — "
            "testing whether arousal during accumulation amplifies or "
            "attenuates the repetition bias in the drift. No main effect of "
            "pupil in drift (consistent with logistic regression)."
        ),
        "tier": 3,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + pupil:choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_pupil:choice_prev1", "z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-x0_Pupil-v_c0": {
        "description": (
            "Pupil model with additional drift bias term. Previous choice "
            "shifts z AND adds a drift constant (c0) AND pupil:choice_prev1 "
            "modulates drift. Tests whether pupil explains residual drift "
            "history effects beyond the additive choice bias. More complex "
            "than the primary pupil model — include to check if c0 and "
            "pupil:choice_prev1 are redundant."
        ),
        "tier": 3,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain + loss + choice_prev1 + pupil:choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [
            "v_choice_prev1",
            "v_pupil:choice_prev1",
            "z_choice_prev1",
        ],
        **DEFAULT_MODEL_PARAMS,
    },

}


# =============================================================================
# MODEL COMPARISON ORDER
# =============================================================================
#
# Run in this order to preserve the nesting argument for BIC:
#
# Step 1 — Identify previous choice mechanism (Tier 0 vs Tier 1)
#   HDDM_Collapse
#   HDDM_Collapse_Lag1-x0       ← expected winner
#   HDDM_Collapse_Lag1-c0
#   HDDM_Collapse_Lag1-stim
#   HDDM_Collapse_Lag1-ndt
#
# Step 2 — Test whether mechanisms combine (Tier 2, only if needed)
#   HDDM_Collapse_Lag1-x0_c0
#   HDDM_Collapse_Lag1-x0_stim
#
# Step 3 — Add pupil to winning model (Tier 3)
#   HDDM_Collapse_Lag1-x0_Pupil-v0   ← pupil null (main effect only)
#   HDDM_Collapse_Lag1-x0_Pupil-v    ← primary pupil model
#   HDDM_Collapse_Lag1-x0_Pupil-v_c0 ← pupil + drift bias (complexity check)
#
# BIC NOTE:
#   Because pupil models are built ON TOP of the winning Tier 1 model,
#   BIC is comparing a model with k parameters vs k+1 parameters. If
#   Pupil-v beats the z-only model under BIC, the improvement is genuine
#   (BIC penalizes the extra pupil:choice_prev1 term). The logistic
#   regression finding pre-registers the direction of the effect.
#
# =============================================================================

MODEL_COMPARISON_ORDER = [
    # Tier 0
    "HDDM_Collapse",
    # Tier 1
    "HDDM_Collapse_Lag1-x0",
    "HDDM_Collapse_Lag1-c0",
    "HDDM_Collapse_Lag1-stim",
    "HDDM_Collapse_Lag1-ndt",
    # Tier 2
    "HDDM_Collapse_Lag1-x0_c0",
    "HDDM_Collapse_Lag1-x0_stim",
    # Tier 3
    "HDDM_Collapse_Lag1-x0_Pupil-v0",
    "HDDM_Collapse_Lag1-x0_Pupil-v",
    "HDDM_Collapse_Lag1-x0_Pupil-v_c0",
]


# =============================================================================
# PARAMETER COUNT TABLE (for BIC penalty accounting)
# =============================================================================
#
# Each row: (model_name, n_group_params_added_over_baseline)
# Baseline has: v_1, v_gain, v_loss, a, t, z, theta  →  7 group-level params
# Each regressor added to group_only_nodes adds 1 group param (mean + variance
# if hierarchical, but group_only means only the mean is estimated).
#
# Use this to verify your BIC penalty is proportionate and to respond to
# reviewer questions about parameter count.
#
# =============================================================================
# MODEL ACCESSOR FUNCTIONS
# =============================================================================


def get_available_models():
    """Return list of all registered model IDs."""
    return list(MODEL_REGISTRY.keys())


def get_model_description(model_id: str) -> str:
    """Return the description string for a model."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    return MODEL_REGISTRY[model_id]["description"]


def get_model_config(model_id: str) -> dict:
    """Return the HDDM config dict for a model (without metadata keys)."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    config = MODEL_REGISTRY[model_id].copy()
    config.pop("description", None)
    config.pop("tier", None)
    return config


PARAM_DELTA_FROM_BASELINE = {
    "HDDM_Collapse":                        0,
    "HDDM_Collapse_Lag1-x0":               1,   # z_choice_prev1
    "HDDM_Collapse_Lag1-c0":               1,   # v_choice_prev1
    "HDDM_Collapse_Lag1-stim":             2,   # v_gain:prev, v_loss:prev
    "HDDM_Collapse_Lag1-ndt":              1,   # t_repeat
    "HDDM_Collapse_Lag1-x0_c0":           2,   # z_prev, v_prev
    "HDDM_Collapse_Lag1-x0_stim":         3,   # z_prev, v_gain:prev, v_loss:prev
    "HDDM_Collapse_Lag1-x0_Pupil-v0":     2,   # z_prev, v_pupil
    "HDDM_Collapse_Lag1-x0_Pupil-v":      2,   # z_prev, v_pupil:prev
    "HDDM_Collapse_Lag1-x0_Pupil-v_c0":   3,   # z_prev, v_prev, v_pupil:prev
}