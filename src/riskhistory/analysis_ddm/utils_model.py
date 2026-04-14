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

MODEL_REGISTRY = {

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
    "HDDM_Collapse_Pupil_Lag1": {
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
    "HDDM_Collapse_Pupil_Lag1-z0": {
        "tier": 4,
        "reg_descr": [
            {
                "model": "v ~ 1 + gain_centered + loss_centered + choice_prev1",
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
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain_centered + loss_centered + choice_prev1 "
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
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain_centered + loss_centered + choice_prev1 "
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
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain_centered + loss_centered + choice_prev1 "
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

    "HDDM_Collapse_Pupil_Lag1-z0_v0_stim": {
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain_centered + loss_centered + choice_prev1 "
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
        "reg_descr": [
            {
                "model": (
                    "v ~ 1 + gain_centered + loss_centered + choice_prev1 "
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
# MODEL ACCESSOR FUNCTIONS
# =============================================================================

def get_available_models() -> list:
    """Return list of all registered model IDs."""
    return list(MODEL_REGISTRY.keys())