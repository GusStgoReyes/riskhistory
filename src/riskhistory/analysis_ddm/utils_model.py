"""Model specification utilities for HDDM fitting."""
import numpy as np


def z_link_func(x):
    """Sigmoid link function for z (starting point bias) parameter."""
    return 1 / (1 + np.exp(-x))


# Default model parameters for HDDMnnRegressor
DEFAULT_MODEL_PARAMS = {
    "model": "angle",
    "informative": False,
    "is_group_model": True,
    "group_only_regressors": False,
    "include": ["v", "a", "t", "z", "theta"],
}

# Registry of available models and their specifications
MODEL_REGISTRY = {
    "HDDM_Collapse": {
        "description": "Baseline model: drift rate depends on gain/loss",
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
        ],
        "group_only_nodes": [],
        "model": "angle",
        "informative": False,
        "is_group_model": True,
        "group_only_regressors": False,
        "include": ["v", "a", "t", "z", "theta"],
    },
    "HDDM_Collapse_Lag1-c0": {
        "description": "Drift rate with lag-1 choice history",
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss + choice_prev1", "link_func": lambda x: x},
        ],
        "group_only_nodes": ["v_choice_prev1"],
        "model": "angle",
        "informative": False,
        "is_group_model": True,
        "group_only_regressors": False,
        "include": ["v", "a", "t", "z", "theta"],
    },
    "HDDM_Collapse_Lag1-x0": {
        "description": "Starting point bias with lag-1 choice history",
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": ["z_choice_prev1"],
        "model": "angle",
        "informative": False,
        "is_group_model": True,
        "group_only_regressors": False,
        "include": ["v", "a", "t", "z", "theta"],
    },
    "HDDM_Collapse_Lag1-x0_c0": {
        "description": "Both drift and starting point with lag-1 choice",
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss + choice_prev1", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1", "link_func": z_link_func},
        ],
        "group_only_nodes": ["v_choice_prev1", "z_choice_prev1"],
        "model": "angle",
        "informative": False,
        "is_group_model": True,
        "group_only_regressors": False,
        "include": ["v", "a", "t", "z", "theta"],
    },
    "HDDM_Collapse_Lag3-x0": {
        "description": "Starting point bias with lag-1,2,3 choice history",
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
            {"model": "z ~ 1 + choice_prev1 + choice_prev2 + choice_prev3", "link_func": z_link_func},
        ],
        "group_only_nodes": ["z_choice_prev1", "z_choice_prev2", "z_choice_prev3"],
        "model": "angle",
        "informative": False,
        "is_group_model": True,
        "group_only_regressors": False,
        "include": ["v", "a", "t", "z", "theta"],
    },
    "HDDM_Collapse_Lag1-ndt": {
        "description": "Non-decision time with repeat behavior",
        "reg_descr": [
            {"model": "v ~ 1 + gain + loss", "link_func": lambda x: x},
            {"model": "t ~ 1 + repeat", "link_func": lambda x: x},
        ],
        "group_only_nodes": ["t_repeat"],
        "model": "angle",
        "informative": False,
        "is_group_model": True,
        "group_only_regressors": False,
        "include": ["v", "a", "t", "z", "theta"],
    },
}


def get_available_models() -> list:
    """Return list of available model IDs."""
    return list(MODEL_REGISTRY.keys())


def get_model_description(model_id: str) -> str:
    """Get the description for a model ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model ID '{model_id}' not recognized. Available: {get_available_models()}")
    return MODEL_REGISTRY[model_id]["description"]


def get_reg_descr(model_id: str) -> list:
    """Get the regression specification for a model ID.

    Parameters
    ----------
    model_id : str
        The model identifier.

    Returns
    -------
    list
        List of regression specification dictionaries.

    Raises
    ------
    ValueError
        If model_id is not recognized.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model ID '{model_id}' not recognized. Available: {get_available_models()}")
    return MODEL_REGISTRY[model_id]["reg_descr"]


def get_group_only_nodes(model_id: str) -> list:
    """Get the group-only nodes for a model ID.

    Parameters
    ----------
    model_id : str
        The model identifier.

    Returns
    -------
    list
        List of parameter names that should be group-only.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model ID '{model_id}' not recognized. Available: {get_available_models()}")
    return MODEL_REGISTRY[model_id]["group_only_nodes"]


def get_model_config(model_id: str) -> dict:
    """Get the full configuration for a model ID.

    Parameters
    ----------
    model_id : str
        The model identifier.

    Returns
    -------
    dict
        Dictionary with keys: reg_descr, group_only_nodes, model, informative,
        is_group_model, group_only_regressors, include.

    Raises
    ------
    ValueError
        If model_id is not recognized.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model ID '{model_id}' not recognized. Available: {get_available_models()}")

    config = MODEL_REGISTRY[model_id].copy()
    # Remove description from config (not needed for model fitting)
    config.pop("description", None)
    return config
