# =============================================================================
# Pupil Beta Extraction Pipeline
# =============================================================================
# Extracts trial-by-trial pupil betas for all subjects using a GLM approach.
# Each trial's regressor is a delta function convolved with a Pupil Response
# Function (PRF).
#
# PRF modes:
#   - subject: Fit PRF separately for each subject (default)
#   - dataset: Fit a single PRF on the grand average across all subjects
#
# Usage:
#   uv run python -m riskhistory.setup_pupilbetas.run_setup_pupilbetas --dataset sheng2020
#   uv run python -m riskhistory.setup_pupilbetas.run_setup_pupilbetas --dataset narps
#   uv run python -m riskhistory.setup_pupilbetas.run_setup_pupilbetas --dataset sheng2020 --prf-mode dataset
#   uv run python -m riskhistory.setup_pupilbetas.run_setup_pupilbetas --test
# =============================================================================

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional

from riskhistory.config import results_pth
from .utils_io import (
    load_start_timings,
    get_valid_subjects,
)
from .utils_prf import fit_dataset_prf
from .utils_glm import fit_prf_and_extract_betas, extract_betas_with_prf
from .utils_plot import plot_subject_diagnostics, plot_prf_fit


def run_extraction(
    dataset: str,
    subjects: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    test_mode: bool = False,
    prf_mode: str = "subject"
) -> None:
    """
    Run pupil beta extraction for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name ("sheng2020" or "equalIndifference" or "equalRange")
    subjects : list, optional
        Subjects to process. If None, process all valid subjects.
    output_dir : Path, optional
        Output directory. If None, uses results/setup_pupilbetas/
    test_mode : bool
        If True, generate diagnostic plots for each subject
    prf_mode : str
        PRF estimation mode: "subject" (per-subject) or "dataset" (grand average)
    """
    print("=" * 60)
    print(f"Pupil Beta Extraction: {dataset}")
    print(f"PRF mode: {prf_mode}")
    print("=" * 60)

    # Setup output directory
    if output_dir is None:
        output_dir = results_pth / "setup_pupilbetas"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames include prf_mode for dataset mode
    if prf_mode == "dataset":
        betas_path = output_dir / f"{dataset}_pupil_betas_datasetPRF.csv"
        prf_path = output_dir / f"{dataset}_prf_parameters_datasetPRF.csv"
    else:
        betas_path = output_dir / f"{dataset}_pupil_betas.csv"
        prf_path = output_dir / f"{dataset}_prf_parameters.csv"

    # Setup diagnostics directory for test mode
    diagnostics_dir = output_dir / "diagnostics" / dataset
    if test_mode:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Load timing data
    print(f"\nLoading timing data...")
    timings_df = load_start_timings(dataset)
    print(f"  Loaded {len(timings_df)} trial timings")

    # Subjects to process
    if subjects is None:
        subjects = get_valid_subjects(dataset)
    print(f"  Processing {len(subjects)} subjects")

    # Dataset-level PRF fitting
    dataset_prf_params = None
    dataset_prf_info = None
    if prf_mode == "dataset":
        print(f"\nFitting dataset-level PRF...")
        dataset_prf_params, dataset_prf_info = fit_dataset_prf(
            dataset, timings_df, subjects,
            epoch_start=0.0, epoch_end=5.0, baseline_end=0.5
        )

        # Save dataset PRF plot
        if test_mode:
            prf_plot_path = diagnostics_dir / f"{dataset}_dataset_prf_fit.png"
            # Add required fields for plotting
            dataset_prf_info['sub'] = 'all'
            dataset_prf_info['dataset'] = dataset
            plot_prf_fit(dataset_prf_info, output_path=prf_plot_path, show=False)
            print(f"  Saved PRF plot: {prf_plot_path}")

    # Process each subject
    all_betas = []
    all_prf_params = []
    n_subjects = len(subjects)

    print("\nExtracting pupil betas...")
    for i, sub_id in enumerate(subjects):
        print(f"  [{i+1}/{n_subjects}] Subject {sub_id}...", end=" ", flush=True)

        if prf_mode == "dataset":
            # Use pre-fitted dataset PRF
            betas_df, prf_info = extract_betas_with_prf(
                sub_id, dataset, timings_df, dataset_prf_params,
                prf_duration=10.0, return_diagnostics=test_mode
            )
            prf_info['prf_mode'] = 'dataset'
        else:
            # Fit subject-specific PRF
            betas_df, prf_info = fit_prf_and_extract_betas(
                sub_id, dataset, timings_df,
                epoch_start=0.0, epoch_end=5.0, prf_duration=10.0,
                return_diagnostics=test_mode
            )
            prf_info['prf_mode'] = 'subject'

        all_betas.append(betas_df)

        # Store PRF params (without diagnostics for CSV)
        prf_row = {
            'sub': prf_info['sub'],
            'dataset': prf_info['dataset'],
            'source_dataset': prf_info['source_dataset'],
            'amplitude': prf_info['amplitude'],
            'a2': prf_info['a2'],
            'tau_rise_c': prf_info['tau_rise_c'],
            'tau_decay_c': prf_info['tau_decay_c'],
            'tau_rise_d': prf_info['tau_rise_d'],
            'tau_decay_d': prf_info['tau_decay_d'],
            'prf_mode': prf_info['prf_mode'],
        }
        # Only add n_epochs_for_fit for subject mode
        if prf_mode == "subject":
            prf_row['n_epochs_for_fit'] = prf_info.get('n_epochs_for_fit', None)
        all_prf_params.append(prf_row)

        # Generate diagnostic plots in test mode (only for subject mode)
        if test_mode and prf_mode == "subject":
            plot_subject_diagnostics(prf_info, diagnostics_dir, show=False)
            print(f"done ({len(betas_df)} trials, plots saved)")
        else:
            print(f"done ({len(betas_df)} trials)")

    # Combine and save results
    betas_combined = pd.concat(all_betas, ignore_index=True)
    prf_params_df = pd.DataFrame(all_prf_params)

    print(f"\nSaving results...")
    print(f"  Betas: {betas_path}")
    betas_combined.to_csv(betas_path, index=False)

    print(f"  PRF parameters: {prf_path}")
    prf_params_df.to_csv(prf_path, index=False)

    if test_mode:
        print(f"  Diagnostic plots: {diagnostics_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PRF mode: {prf_mode}")
    if prf_mode == "dataset" and dataset_prf_info:
        print(f"Dataset PRF: amp={dataset_prf_info['amplitude']:.3f}, "
              f"a2={dataset_prf_info['a2']:.3f}, "
              f"tau_rise_c={dataset_prf_info['tau_rise_c']:.3f}, "
              f"tau_decay_c={dataset_prf_info['tau_decay_c']:.3f}, "
              f"tau_rise_d={dataset_prf_info['tau_rise_d']:.3f}, "
              f"tau_decay_d={dataset_prf_info['tau_decay_d']:.3f}")
    print(f"Subjects processed: {len(all_prf_params)}")
    print(f"Total betas: {len(betas_combined)}")
    print(f"Beta range: [{betas_combined['pupil_beta'].min():.3f}, "
          f"{betas_combined['pupil_beta'].max():.3f}]")
    print(f"Mean beta: {betas_combined['pupil_beta'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pupil betas via GLM"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["sheng2020", "narps"],
        default="sheng2020",
        help="Dataset to process (default: sheng2020)"
    )
    parser.add_argument(
        "--subjects", "-s",
        nargs="+",
        type=int,
        default=None,
        help="Subject IDs to process (default: all valid)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: results/setup_pupilbetas/)"
    )
    parser.add_argument(
        "--prf-mode", "-p",
        type=str,
        choices=["subject", "dataset"],
        default="subject",
        help="PRF estimation mode: 'subject' for per-subject PRF, "
             "'dataset' for grand average PRF (default: subject)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process first 3 subjects with diagnostic plots"
    )

    args = parser.parse_args()

    subjects = args.subjects
    test_mode = args.test

    if args.test and subjects is None:
        valid = get_valid_subjects(args.dataset)
        subjects = valid[:3]
        print(f"TEST MODE: Processing subjects {subjects} with diagnostic plots")

    output_dir = Path(args.output) if args.output else None

    run_extraction(
        dataset=args.dataset,
        subjects=subjects,
        output_dir=output_dir,
        test_mode=test_mode,
        prf_mode=args.prf_mode
    )


if __name__ == "__main__":
    main()
