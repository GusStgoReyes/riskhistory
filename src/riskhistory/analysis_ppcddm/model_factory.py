from riskhistory.analysis_ppcddm.ddm_model import PTDDM
from riskhistory.analysis_ppcddm.functions_ddm import (
    drift_linear,
    bound,
    startingposition,
    nondecision,
    noise
)

from riskhistory.analysis_ppcddm.parameters import FullDDMParameters
                                 
# Parameters for the numerical solution of the DDM
dx=0.002
dt=0.002 
T_dur=4

def get_model(model_id):
    """
    Factory function to get the model instance based on the model ID.
    Define your own model here!
    model_id : str
                Model ID to instantiate.
    """
    if model_id == "HDDM_Collapse":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
            "v_loss", 
            "v_Intercept",
            "a",
            "theta",
            "z",
            "ter"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(1,7)]
        )

    if model_id == "HDDM_Collapse_Lag1-c0":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
            "v_loss", 
            "v_Intercept",
            "a",
            "theta",
            "z",
            "ter",
            "v_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2,7)]
        )
    elif model_id == "HDDM_Collapse_Lag1-x0":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
            "v_loss", 
            "v_Intercept",
            "a",
            "theta",
            "z",
            "ter",
            "z_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2,7)]
        )
        
    elif model_id == "HDDM_Collapse_Lag1-stim":
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
            "v_loss", 
            "v_Intercept",
            "a",
            "theta",
            "z",
            "ter",
            "v_loss_choice_prev1",
            "v_gain_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2,7)]
        )
    
    if "Tanh" in model_id:
        pass
    else:
        drift = drift_linear

    return PTDDM(model_id,
                drift,
                bound,
                startingposition,
                nondecision,
                noise,
                parameters = parameters,
                dx=dx,
                dt=dt,
                T_dur=T_dur)
