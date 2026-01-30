# =============================================================================
# Pupil Beta Extraction: Plotting Utilities
# =============================================================================
# Functions for visualizing PRF fits and GLM diagnostics.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .utils_prf import biphasic_prf


def plot_prf_fit(
    prf_info: dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot the mean pupil response and fitted PRF.

    Parameters
    ----------
    prf_info : dict
        PRF info dictionary with 'time_vec', 'mean_pupil', 'fitted_prf'
        and PRF parameters
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    time_vec = prf_info['time_vec']
    mean_pupil = prf_info['mean_pupil']
    fitted_prf = prf_info['fitted_prf']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean pupil response
    ax.plot(time_vec, mean_pupil, 'b-', linewidth=2, label='Mean pupil response')

    # Plot fitted PRF
    ax.plot(time_vec, fitted_prf, 'r--', linewidth=2, label='Fitted PRF')

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Labels
    ax.set_xlabel('Time from trial onset (s)', fontsize=12)
    ax.set_ylabel('Pupil size (baseline-corrected)', fontsize=12)
    ax.set_title(
        f"Subject {prf_info['sub']} ({prf_info['dataset']})\n"
        f"PRF: amp={prf_info['amplitude']:.3f}, a2={prf_info['a2']:.3f}, "
        f"tau_rise_c={prf_info['tau_rise_c']:.3f}, tau_decay_c={prf_info['tau_decay_c']:.3f}, "
        f"tau_rise_d={prf_info['tau_rise_d']:.3f}, tau_decay_d={prf_info['tau_decay_d']:.3f}",
        fontsize=9
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_residuals(
    prf_info: dict,
    run_idx: int = 0,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot GLM residuals for a specific run.

    Parameters
    ----------
    prf_info : dict
        PRF info dictionary with 'run_diagnostics'
    run_idx : int
        Index of the run to plot (0-based)
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    diagnostics = prf_info['run_diagnostics'][run_idx]
    residuals = diagnostics['residuals']
    y_predicted = diagnostics['y_predicted']
    y_observed = diagnostics['y_observed']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time series: observed vs predicted (first 5000 samples)
    ax1 = axes[0, 0]
    n_plot = min(5000, len(y_observed))
    ax1.plot(y_observed[:n_plot], 'b-', alpha=0.7, linewidth=0.5, label='Observed')
    ax1.plot(y_predicted[:n_plot], 'r-', alpha=0.7, linewidth=0.5, label='Predicted')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Pupil size')
    ax1.set_title(f"Run {diagnostics['run']}: Observed vs Predicted (first {n_plot} samples)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals over time
    ax2 = axes[0, 1]
    ax2.plot(residuals[:n_plot], 'k-', alpha=0.5, linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Residual')
    ax2.set_title(f'Residuals over time (first {n_plot} samples)')
    ax2.grid(True, alpha=0.3)

    # 3. Residual histogram
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Residual distribution (mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f})')
    ax3.grid(True, alpha=0.3)

    # 4. Predicted vs Observed scatter
    ax4 = axes[1, 1]
    # Subsample for scatter plot if too many points
    n_scatter = min(2000, len(y_observed))
    idx = np.random.choice(len(y_observed), n_scatter, replace=False)
    ax4.scatter(y_observed[idx], y_predicted[idx], alpha=0.3, s=5)

    # Identity line
    min_val = min(y_observed.min(), y_predicted.min())
    max_val = max(y_observed.max(), y_predicted.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Identity')

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    ax4.set_xlabel('Observed')
    ax4.set_ylabel('Predicted')
    ax4.set_title(f'Predicted vs Observed (R² = {r_squared:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"Subject {prf_info['sub']} ({prf_info['dataset']}) - Run {diagnostics['run']}",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_subject_diagnostics(
    prf_info: dict,
    output_dir: Path,
    show: bool = False
) -> None:
    """
    Generate all diagnostic plots for a subject.

    Creates:
    - PRF fit plot
    - Residual plots for each run

    Parameters
    ----------
    prf_info : dict
        PRF info dictionary with diagnostics
    output_dir : Path
        Directory to save plots
    show : bool
        Whether to display the figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sub_id = prf_info['sub']
    dataset = prf_info['source_dataset']

    # Plot PRF fit
    prf_path = output_dir / f"sub-{sub_id:03d}_{dataset}_prf_fit.png"
    fig_prf = plot_prf_fit(prf_info, output_path=prf_path, show=show)
    plt.close(fig_prf)

    # Plot residuals for each run
    if 'run_diagnostics' in prf_info:
        for run_idx, run_diag in enumerate(prf_info['run_diagnostics']):
            run_id = run_diag['run']
            resid_path = output_dir / f"sub-{sub_id:03d}_{dataset}_run-{run_id}_residuals.png"
            fig_resid = plot_residuals(prf_info, run_idx=run_idx, output_path=resid_path, show=show)
            plt.close(fig_resid)
