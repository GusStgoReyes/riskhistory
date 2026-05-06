from riskhistory.analysis_ppcddm.ddm_model import PTDDM
from riskhistory.analysis_ppcddm.functions_ddm import (
    drift_linear,
    bound,
    startingposition, startingposition_sigmoid,
    nondecision,
    noise
)

from riskhistory.analysis_ppcddm.parameters import FullDDMParameters

# Parameters for the numerical solution of the DDM
dx = 0.002
dt = 0.002
T_dur = 4


def get_model(model_id):
    """
    Factory function to get the model instance based on the model ID.

    model_id : str
        One of: HDDM_Base, HDDM_Base_Lag1-drift, HDDM_Base_Lag1-stp,
                HDDM_Base_Lag1-stim, HDDM_Base_Lag1-full
    """
    if model_id == "HDDM_Base":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
             "v_loss",
             "v_Intercept",
             "a",
             "z",
             "ter"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(1, 7)]
        )

    elif model_id == "HDDM_Base_Lag1-drift":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
             "v_loss",
             "v_Intercept",
             "a",
             "z",
             "ter",
             "v_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )

    elif model_id == "HDDM_Base_Lag1-stp":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
             "v_loss",
             "v_Intercept",
             "a",
             "z",
             "ter",
             "z_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )

    elif model_id == "HDDM_Base_Lag1-stim":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
             "v_loss",
             "v_Intercept",
             "a",
             "z",
             "ter",
             "v_gain_choice_prev1",
             "v_loss_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )

    elif model_id == "HDDM_Base_Lag1-full":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
             "v_loss",
             "v_Intercept",
             "a",
             "z",
             "ter",
             "v_choice_prev1",
             "v_gain_choice_prev1",
             "v_loss_choice_prev1",
             "z_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )

    else:
        raise ValueError(f"Unknown model_id: '{model_id}'")

    drift = drift_linear

    # stp and full use a logistic link for z (matching HDDM's z_link_func).
    # Base, drift, and stim have no z regression: z from HDDM is already the
    # actual starting-point fraction in (0, 1), so linear scaling suffices.
    if model_id in ("HDDM_Base_Lag1-stp", "HDDM_Base_Lag1-full"):
        startingposition_fn = startingposition_sigmoid
    else:
        startingposition_fn = startingposition

    return PTDDM(model_id,
                 drift,
                 bound,
                 startingposition_fn,
                 nondecision,
                 noise,
                 parameters=parameters,
                 dx=dx,
                 dt=dt,
                 T_dur=T_dur)
