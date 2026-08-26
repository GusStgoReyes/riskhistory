from riskhistory.analysis_ppcddm.ddm_model import PTDDM
from riskhistory.analysis_ppcddm.functions_ddm import (
    drift_linear, drift_linear_pupil,
    bound, bound_pupil,
    startingposition, startingposition_sigmoid, startingposition_sigmoid_pupil,
    nondecision, nondecision_pupil,
    noise
)

from riskhistory.analysis_ppcddm.parameters import FullDDMParameters, PupilDDMParameters

# Parameters for the numerical solution of the DDM
dx = 0.002
dt = 0.002
T_dur = 4


def get_model(model_id):
    """
    Factory function to get the model instance based on the model ID.

    model_id : str
        One of: HDDM_Base, HDDM_Base_Lag1-drift, HDDM_Base_Lag1-stp,
                HDDM_Base_Lag1-stim, HDDM_Base_Lag1-full,
                HDDM_Base_Lag1-stpdrift,
                HDDM_Base_Lag1-stpdrift_Pupil,
                HDDM_Base_Lag1-stpdrift_Pupil_stpdrift,
                HDDM_Base_Lag1-stpdrift_Pupil_valuation,
                HDDM_Base_Lag1-stpdrift_Pupil_boundary,
                HDDM_Base_Lag1-stpdrift_Pupil_ndt,
                HDDM_Base_Lag1-stpdrift_Pupil_drift,
                HDDM_Base_Lag1-stpdrift_Pupil_stp,
                HDDM_Base_Lag1-drift_Pupil,
                HDDM_Base_Lag1-drift_Pupil_drift,
                HDDM_Base_Lag1-stp_Pupil,
                HDDM_Base_Lag1-stp_Pupil_stp
    """
    # Defaults; individual branches below override whichever of these need
    # a pupil-aware variant for that arousal mechanism.
    drift = drift_linear
    bound_fn = bound
    nondecision_fn = nondecision
    startingposition_fn = startingposition

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

    elif model_id in ("HDDM_Base_Lag1-drift", "HDDM_Base_Lag1-drift_Pupil"):
        # HDDM_Base_Lag1-drift_Pupil is structurally identical -- no pupil
        # interaction terms -- it's just fit/simulated on the pupil-clean
        # trial population (routed via PUPIL_MODEL_MARKER in
        # run_analysis_ppcddm.py) so it's a fair PPC/ELPD baseline for
        # HDDM_Base_Lag1-drift_Pupil_drift.
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

    elif model_id in ("HDDM_Base_Lag1-stp", "HDDM_Base_Lag1-stp_Pupil"):
        # HDDM_Base_Lag1-stp_Pupil is structurally identical -- no pupil
        # interaction terms -- it's just fit/simulated on the pupil-clean
        # trial population (routed via PUPIL_MODEL_MARKER in
        # run_analysis_ppcddm.py) so it's a fair PPC/ELPD baseline for
        # HDDM_Base_Lag1-stp_Pupil_stp.
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
        # stp and full use a logistic link for z (matching HDDM's
        # z_link_func). Base, drift, and stim have no z regression: z from
        # HDDM is already the actual starting-point fraction in (0, 1), so
        # linear scaling suffices.
        startingposition_fn = startingposition_sigmoid

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
        startingposition_fn = startingposition_sigmoid

    elif model_id in ("HDDM_Base_Lag1-stpdrift", "HDDM_Base_Lag1-stpdrift_Pupil"):
        # HDDM_Base_Lag1-stpdrift_Pupil is structurally identical -- no
        # pupil interaction terms, "pupil" isn't a model condition at all --
        # it's just fit/simulated on the pupil-clean trial population
        # (routed via PUPIL_MODEL_MARKER in run_analysis_ppcddm.py) so it's
        # a fair PPC/ELPD baseline for the HDDM_Base_Lag1-stpdrift_Pupil_*
        # mechanism models, which all need that same restricted population.
        parameters = FullDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain",
             "v_loss",
             "v_Intercept",
             "a",
             "z",
             "ter",
             "v_choice_prev1",
             "z_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        startingposition_fn = startingposition_sigmoid

    # -----------------------------------------------------------------------
    # Pupil-arousal models
    # -----------------------------------------------------------------------
    elif model_id == "HDDM_Base_Lag1-stpdrift_Pupil_stpdrift":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z", "ter",
             "v_choice_prev1", "z_choice_prev1",
             "v_pupil", "v_choice_prev1_pupil",
             "z_pupil", "z_choice_prev1_pupil"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        drift = drift_linear_pupil
        startingposition_fn = startingposition_sigmoid_pupil

    elif model_id == "HDDM_Base_Lag1-stpdrift_Pupil_valuation":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z", "ter",
             "v_choice_prev1", "z_choice_prev1",
             "v_pupil", "v_gain_pupil", "v_loss_pupil"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        drift = drift_linear_pupil
        startingposition_fn = startingposition_sigmoid

    elif model_id == "HDDM_Base_Lag1-stpdrift_Pupil_boundary":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a_Intercept", "a_pupil",
             "z", "ter", "v_choice_prev1", "z_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        bound_fn = bound_pupil
        startingposition_fn = startingposition_sigmoid

    elif model_id == "HDDM_Base_Lag1-stpdrift_Pupil_ndt":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z",
             "t_Intercept", "t_pupil", "v_choice_prev1", "z_choice_prev1"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        nondecision_fn = nondecision_pupil
        startingposition_fn = startingposition_sigmoid

    elif model_id == "HDDM_Base_Lag1-stpdrift_Pupil_drift":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z", "ter",
             "v_choice_prev1", "z_choice_prev1", "v_choice_prev1_pupil", "v_pupil"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        drift = drift_linear_pupil
        startingposition_fn = startingposition_sigmoid

    elif model_id == "HDDM_Base_Lag1-stpdrift_Pupil_stp":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z", "ter",
             "v_choice_prev1", "z_choice_prev1", "z_choice_prev1_pupil", "z_pupil"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        startingposition_fn = startingposition_sigmoid_pupil

    # -----------------------------------------------------------------------
    # Single-mechanism pupil models: built on the drift-only / stp-only base
    # (no combined choice-history structure, unlike the
    # HDDM_Base_Lag1-stpdrift_Pupil_* models above), with pupil interacting
    # with just that one mechanism.
    # -----------------------------------------------------------------------
    elif model_id == "HDDM_Base_Lag1-drift_Pupil_drift":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z", "ter",
             "v_choice_prev1", "v_choice_prev1_pupil", "v_pupil"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        drift = drift_linear_pupil

    elif model_id == "HDDM_Base_Lag1-stp_Pupil_stp":
        parameters = PupilDDMParameters()
        parameters.set_parameters_to_fit(
            ["v_gain", "v_loss", "v_Intercept", "a", "z", "ter",
             "z_choice_prev1", "z_choice_prev1_pupil", "z_pupil"]
        )
        parameters.set_conditions_constant(
            [f"choice_prev{i}" for i in range(2, 7)]
        )
        startingposition_fn = startingposition_sigmoid_pupil

    else:
        raise ValueError(f"Unknown model_id: '{model_id}'")

    return PTDDM(model_id,
                 drift,
                 bound_fn,
                 startingposition_fn,
                 nondecision_fn,
                 noise,
                 parameters=parameters,
                 dx=dx,
                 dt=dt,
                 T_dur=T_dur)
