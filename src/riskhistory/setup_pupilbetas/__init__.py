# =============================================================================
# Pupil Beta Extraction Module
# =============================================================================
"""
Extract trial-by-trial pupil betas using a GLM approach.

Each trial's regressor is a delta function convolved with a subject-specific
Pupil Response Function (PRF).

Modules:
    utils_io: Data loading (timeseries, timings, subject discovery, exclusions)
    utils_prf: PRF functions and fitting
    utils_glm: GLM design matrix and beta extraction
    utils_plot: Visualization of PRF fits and residuals
"""

from .utils_io import (
    DATASET_CONFIG,
    get_analysis_dataset,
    remove_outliers,
    load_timeseries,
    load_start_timings,
    load_subject_exclusions,
    get_subject_run_timings,
    get_valid_subjects,
    get_subject_runs,
    estimate_sampling_rate,
)

from .utils_prf import (
    biphasic_prf,
    dilation_component,
    constriction_component,
    extract_epoch,
    compute_mean_pupil_run,
    compute_mean_pupil_subject,
    compute_mean_pupil_dataset,
    fit_prf,
    fit_dataset_prf,
    create_prf_kernel,
    create_dilation_kernel,
    create_constriction_kernel,
)

from .utils_glm import (
    GLMResult,
    create_control_regressors,
    create_trial_regressors,
    create_constriction_regressor,
    fit_glm_with_controls,
    fit_glm_run,
    extract_betas_subject,
    extract_betas_with_prf,
    fit_prf_and_extract_betas,
)

from .utils_plot import (
    plot_prf_fit,
    plot_residuals,
    plot_subject_diagnostics,
)

__all__ = [
    # Config
    'DATASET_CONFIG',
    'get_analysis_dataset',
    # I/O
    'remove_outliers',
    'load_timeseries',
    'load_start_timings',
    'load_subject_exclusions',
    'get_subject_run_timings',
    'get_valid_subjects',
    'get_subject_runs',
    'estimate_sampling_rate',
    # PRF
    'biphasic_prf',
    'dilation_component',
    'constriction_component',
    'extract_epoch',
    'compute_mean_pupil_run',
    'compute_mean_pupil_subject',
    'compute_mean_pupil_dataset',
    'fit_prf',
    'fit_dataset_prf',
    'create_prf_kernel',
    'create_dilation_kernel',
    'create_constriction_kernel',
    # GLM
    'GLMResult',
    'create_control_regressors',
    'create_trial_regressors',
    'create_constriction_regressor',
    'fit_glm_with_controls',
    'fit_glm_run',
    'extract_betas_subject',
    'extract_betas_with_prf',
    'fit_prf_and_extract_betas',
    # Plotting
    'plot_prf_fit',
    'plot_residuals',
    'plot_subject_diagnostics',
]
