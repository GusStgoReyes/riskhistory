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
    "backend": "HDDMnnRegressor",
    "model": "angle",
    "informative": False,
    "is_group_model": True,
    "group_only_regressors": False,
    "include": ["v", "a", "t", "z", "theta"],
}

DEFAULT_MODEL_PARAMS_NOCOLLAPSE = {
    "backend": "HDDMRegressor",
    "informative": False,
    "is_group_model": True,
    "group_only_regressors": False,
    "include": ["v", "a", "t", "z"],
}

MODEL_REGISTRY = {

    # -------------------------------------------------------------------------
    # Collapsing-boundary models (HDDMnnRegressor, angle kernel, theta included)
    # -------------------------------------------------------------------------

    "HDDM_Collapse": {
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered", "link_func": lambda x: x},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-z0": {
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered",  "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1",      "link_func": z_link_func},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-v0": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-z0_v0": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS,
    },

    # -------------------------------------------------------------------------
    # Non-collapsing models (HDDMRegressor, standard Wiener kernel, no theta)
    # -------------------------------------------------------------------------

    "HDDM_Base": {
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered", "link_func": lambda x: x},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-stp": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered", 
                "link_func": lambda x: x
            },
            {
                "model": "z ~ 1 + choice_prev1",
                "link_func": z_link_func
             },
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-drift": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-stim": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + gain_centered:choice_prev1 + loss_centered:choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-full": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + gain_centered:choice_prev1 + loss_centered:choice_prev1 + choice_prev1",
                "link_func": lambda x: x,
            },
            {
                "model": "z ~ 1 + choice_prev1",
                "link_func": z_link_func,
            }
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    # -------------------------------------------------------------------------
    # Pupil models (fit on the pupil-eligible subject/trial subset -- "Pupil"
    # in the model_id triggers the pupil merge + exclusion in load_data).
    # -------------------------------------------------------------------------

    "HDDM_Base_Pupil": {
        # Same formula as HDDM_Base, refit on the pupil-eligible sample --
        # a reference point for comparing against the pupil-regressor models
        # below without the sample itself being a confound.
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered", "link_func": lambda x: x},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-stp_Pupil": {
        # Same formula as HDDM_Base_Lag1-stp, refit on the pupil-eligible
        # sample -- confirms the starting-point bias still holds in this
        # (slightly smaller) sample before testing whether pupil modulates it.
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-stp_Pupil_stp": {
        # Does pupil moderate the starting-point choice-history bias?
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1 * pupil", "link_func": z_link_func},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-stp_Pupil_drift": {
        # Does pupil moderate value sensitivity (gain/loss weighting) in drift?
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered * pupil + loss_centered * pupil",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Base_Lag1-stp_Pupil_full": {
        # Both of the above: pupil x value sensitivity in drift, and pupil x
        # choice-history bias in starting point.
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered * pupil + loss_centered * pupil",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1 * pupil", "link_func": z_link_func},
        ],
        "group_only_nodes": [],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },
}

# =============================================================================
# MODEL ACCESSOR FUNCTIONS
# =============================================================================

def get_available_models() -> list:
    """Return list of all registered model IDs."""
    return list(MODEL_REGISTRY.keys())

def get_model_config(model_id: str) -> dict:
    """Return the configuration dictionary for a given model ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model ID '{model_id}' not found in registry.")
    return MODEL_REGISTRY[model_id]