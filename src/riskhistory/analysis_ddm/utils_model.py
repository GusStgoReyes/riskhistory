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
        "group_only_nodes": ["z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS,
    },

    "HDDM_Collapse_Lag1-v0": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": ["v_choice_prev1"],
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
        "group_only_nodes": ["v_choice_prev1", "z_choice_prev1"],
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

    "HDDM_Lag1-z0": {
        "reg_descr": [
            {"model": "v ~ 1 + gain_centered + loss_centered", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1",                  "link_func": z_link_func},
        ],
        "group_only_nodes": ["z_choice_prev1"],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Lag1-v0": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
                "link_func": lambda x: x,
            },
        ],
        "group_only_nodes": ["v_choice_prev1"],
        **DEFAULT_MODEL_PARAMS_NOCOLLAPSE,
    },

    "HDDM_Lag1-z0_v0": {
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
                "link_func": lambda x: x,
            },
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_choice_prev1", "z_choice_prev1"],
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