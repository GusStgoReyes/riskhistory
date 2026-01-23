"""
Variance Inflation Factor (VIF) Analysis for GLM Regressors
===========================================================

This script computes VIF for each regressor in your GLM to check for
multicollinearity. High VIF (>10) indicates problematic collinearity.

VIF interpretation:
- VIF = 1: No correlation with other regressors
- VIF < 5: Low multicollinearity (acceptable)
- VIF 5-10: Moderate multicollinearity (caution)
- VIF > 10: High multicollinearity (problematic)

Author: Generated for NARPS mixed gambles analysis
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')


def compute_vif(design_matrix, regressors_of_interest):
    """
    Compute VIF for specified regressors.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        Design matrix with all regressors
    regressors_of_interest : list
        List of regressor names to compute VIF for
        
    Returns
    -------
    pd.DataFrame
        VIF values for each regressor
    """
    # Filter to only regressors of interest that exist
    available_regressors = [r for r in regressors_of_interest if r in design_matrix.columns]
    
    if len(available_regressors) == 0:
        return None
    
    # Extract only these columns
    X = design_matrix[available_regressors].values
    
    # Compute VIF for each regressor
    vif_data = []
    for i, regressor in enumerate(available_regressors):
        try:
            vif = variance_inflation_factor(X, i)
            vif_data.append({
                'Regressor': regressor,
                'VIF': vif
            })
        except Exception as e:
            print(f"  Warning: Could not compute VIF for {regressor}: {e}")
            vif_data.append({
                'Regressor': regressor,
                'VIF': np.nan
            })
    
    return pd.DataFrame(vif_data)


def create_design_matrix_from_events(event_file, 
                                     confounds_file,
                                     n_scans,
                                     tr=1.0,
                                     hrf_model='spm',
                                     high_pass=1./128):
    """
    Create design matrix from event file (matching your fitlins setup).
    
    Parameters
    ----------
    event_file : str
        Path to events TSV file
    confounds_file : str
        Path to confounds file
    n_scans : int
        Number of scans in the run
    tr : float
        Repetition time
    hrf_model : str
        HRF model to use
    high_pass : float
        High-pass filter cutoff
        
    Returns
    -------
    pd.DataFrame
        Design matrix
    """
    # Load events
    events_df = pd.read_csv(event_file, sep='\t')
    
    # Create frame times
    frame_times = np.arange(n_scans) * tr
    
    # Load confounds
    confounds = None
    if confounds_file and os.path.exists(confounds_file):
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        # Select motion parameters
        motion_params = [col for col in confounds_df.columns 
                        if col.startswith(('trans_', 'rot_'))]
        
        if len(motion_params) > 0:
            confounds = confounds_df[motion_params].fillna(0)
    # Create frame times
    frame_times = np.arange(confounds_df.shape[0]) * tr
    # Prepare events for each regressor type
    all_events = []
    
    # The regressors from your fitlins model
    regressors = [
        'strongly_reject',
        'weakly_reject', 
        'weakly_accept',
        'strongly_accept',
        'pos_sv',
        'neg_sv',
        'entropy',
        'RT',
        'switch',
        'repeat'
    ]
    
    for col in regressors:
        if col not in events_df.columns:
            continue
        
        # Get onsets where this regressor is non-zero
        col_events = events_df[events_df[col] != 0].copy()
        
        if len(col_events) == 0:
            continue
        
        for _, row in col_events.iterrows():
            all_events.append({
                'onset': row['onset'],
                'duration': row['duration'],
                'trial_type': col,
                'modulation': row[col]
            })
    
    if len(all_events) == 0:
        return None
    
    events_for_dm = pd.DataFrame(all_events)
    print(events_for_dm["trial_type"].nunique())
    # Create design matrix
    try:
        design_matrix = make_first_level_design_matrix(
            frame_times=frame_times,
            events=events_for_dm,
            hrf_model=hrf_model,
            drift_model='cosine',
            high_pass=high_pass,
            add_regs=confounds,
            add_reg_names=confounds.columns.tolist() if confounds is not None else None
        )
        
        return design_matrix
        
    except Exception as e:
        print(f"  Error creating design matrix: {e}")
        return None


def analyze_subject_run_vif(bids_dir,
                            derivatives_dir,
                            subject,
                            run,
                            regressors_of_interest,
                            tr=1.0):
    """
    Analyze VIF for one subject and run.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS directory
    derivatives_dir : str
        Path to derivatives directory
    subject : str
        Subject ID (e.g., '001')
    run : str
        Run ID (e.g., '01')
    regressors_of_interest : list
        List of regressors to check
    tr : float
        Repetition time
        
    Returns
    -------
    pd.DataFrame
        VIF results with subject and run info
    """
    print(f"\nAnalyzing subject {subject}, run {run}...")
    
    sub_id = f"sub-{subject}"
    
    # File paths
    event_file = os.path.join(bids_dir, sub_id, 'func',
                             f"{sub_id}_task-MGT_run-{run}_events.tsv")
    
    func_file = os.path.join(derivatives_dir, sub_id, 'func',
                            f"{sub_id}_task-MGT_run-{run[1]}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    
    confounds_file = os.path.join(derivatives_dir, sub_id, 'func',
                                 f"{sub_id}_task-MGT_run-{run[1]}_desc-confounds_timeseries.tsv")
    
    # Check files exist
    if not os.path.exists(event_file):
        print(f"  Event file not found: {event_file}")
        return None
    
    if not os.path.exists(func_file):
        print(f"  Functional file not found: {func_file}")
        return None
    
    # Get number of scans
    # import nibabel as nib
    # func_img = nib.load(func_file)
    n_scans = 480 #func_img.shape[-1]
    
    # Create design matrix
    design_matrix = create_design_matrix_from_events(
        event_file=event_file,
        confounds_file=confounds_file if os.path.exists(confounds_file) else None,
        n_scans=n_scans,
        tr=tr
    )
    
    if design_matrix is None:
        print(f"  Could not create design matrix")
        return None
    
    print(f"  Design matrix shape: {design_matrix.shape}")
    print(f"  Available regressors: {[r for r in regressors_of_interest if r in design_matrix.columns]}")
    
    # Compute VIF
    vif_results = compute_vif(design_matrix, regressors_of_interest)
    
    if vif_results is None:
        print(f"  Could not compute VIF")
        return None
    
    # Add subject and run info
    vif_results['Subject'] = subject
    vif_results['Run'] = run
    
    # Print results
    print(f"\n  VIF Results:")
    for _, row in vif_results.iterrows():
        vif_value = row['VIF']
        regressor = row['Regressor']
        
        if pd.isna(vif_value):
            status = "ERROR"
        elif vif_value < 5:
            status = "✓ Good"
        elif vif_value < 10:
            status = "⚠ Moderate"
        else:
            status = "✗ High"
        
        print(f"    {regressor:20s}: VIF = {vif_value:8.2f}  [{status}]")
    
    return vif_results


def analyze_all_subjects(bids_dir,
                         derivatives_dir,
                         subjects,
                         runs=['01', '02', '03', '04'],
                         regressors_of_interest=None,
                         output_dir='vif_results',
                         tr=1.0):
    """
    Analyze VIF for all subjects and runs.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS directory
    derivatives_dir : str
        Path to derivatives directory
    subjects : list
        List of subject IDs
    runs : list
        List of run IDs
    regressors_of_interest : list
        Regressors to analyze
    output_dir : str
        Directory to save results
    tr : float
        Repetition time
        
    Returns
    -------
    pd.DataFrame
        Combined VIF results for all subjects/runs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if regressors_of_interest is None:
        regressors_of_interest = [
            'strongly_reject',
            'weakly_reject',
            'weakly_accept',
            'strongly_accept',
            'pos_sv',
            'neg_sv',
            'entropy',
            'RT',
            'switch',
            'repeat'
        ]
    
    print("="*60)
    print("VIF ANALYSIS FOR GLM REGRESSORS")
    print("="*60)
    print(f"\nRegressors to analyze: {regressors_of_interest}")
    print(f"Number of subjects: {len(subjects)}")
    print(f"Number of runs: {len(runs)}")
    
    all_vif_results = []
    
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Subject {subject}")
        print(f"{'='*60}")
        
        for run in runs:
            vif_results = analyze_subject_run_vif(
                bids_dir=bids_dir,
                derivatives_dir=derivatives_dir,
                subject=subject,
                run=run,
                regressors_of_interest=regressors_of_interest,
                tr=tr
            )
            
            if vif_results is not None:
                all_vif_results.append(vif_results)
    
    if len(all_vif_results) == 0:
        print("\nNo VIF results computed!")
        return None
    
    # Combine all results
    combined_vif = pd.concat(all_vif_results, ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'vif_all_subjects_runs.csv')
    combined_vif.to_csv(csv_path, index=False)
    print(f"\n✓ Saved all results to: {csv_path}")
    
    # Create summary statistics
    create_vif_summary(combined_vif, output_dir)
    
    # Create visualizations
    plot_vif_results(combined_vif, output_dir)
    
    return combined_vif


def create_vif_summary(vif_df, output_dir):
    """
    Create summary statistics of VIF across subjects/runs.
    """
    print("\n" + "="*60)
    print("VIF SUMMARY STATISTICS")
    print("="*60)
    
    summary = vif_df.groupby('Regressor')['VIF'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    # Add interpretation
    summary['Interpretation'] = summary['mean'].apply(
        lambda x: 'Good' if x < 5 else ('Moderate' if x < 10 else 'High')
    )
    
    print("\n" + summary.to_string())
    
    # Save summary
    summary_path = os.path.join(output_dir, 'vif_summary.csv')
    summary.to_csv(summary_path)
    print(f"\n✓ Saved summary to: {summary_path}")
    
    # Check for problems
    print("\n" + "="*60)
    print("VIF INTERPRETATION")
    print("="*60)
    
    high_vif = summary[summary['mean'] > 10]
    moderate_vif = summary[(summary['mean'] >= 5) & (summary['mean'] <= 10)]
    low_vif = summary[summary['mean'] < 5]
    
    if len(high_vif) > 0:
        print("\n⚠ HIGH MULTICOLLINEARITY (VIF > 10):")
        for reg in high_vif.index:
            print(f"  • {reg}: Mean VIF = {summary.loc[reg, 'mean']:.2f}")
        print("\n  These regressors are highly correlated with others.")
        print("  Consider:")
        print("    - Removing one of the correlated regressors")
        print("    - Orthogonalizing the regressors")
        print("    - Using different contrasts")
    
    if len(moderate_vif) > 0:
        print("\n⚠ MODERATE MULTICOLLINEARITY (VIF 5-10):")
        for reg in moderate_vif.index:
            print(f"  • {reg}: Mean VIF = {summary.loc[reg, 'mean']:.2f}")
        print("\n  These regressors show moderate correlation.")
        print("  Proceed with caution, but usually acceptable.")
    
    if len(low_vif) > 0:
        print("\n✓ LOW MULTICOLLINEARITY (VIF < 5):")
        for reg in low_vif.index:
            print(f"  • {reg}: Mean VIF = {summary.loc[reg, 'mean']:.2f}")
    
    # Specific advice for switch/repeat
    if 'switch' in summary.index and 'repeat' in summary.index:
        switch_vif = summary.loc['switch', 'mean']
        repeat_vif = summary.loc['repeat', 'mean']
        
        print("\n" + "="*60)
        print("SWITCH ANALYSIS VALIDITY CHECK")
        print("="*60)
        
        if switch_vif < 10 and repeat_vif < 10:
            print("\n✓ Switch and repeat regressors have acceptable VIF.")
            print("  Your switch analysis is statistically sound!")
        elif switch_vif >= 10 or repeat_vif >= 10:
            print("\n✗ Switch and/or repeat regressors have high VIF.")
            print("  This suggests they are highly correlated with other regressors.")
            print(f"    Switch VIF: {switch_vif:.2f}")
            print(f"    Repeat VIF: {repeat_vif:.2f}")
            print("\n  Potential solutions:")
            print("    1. Remove the response type regressors (strongly/weakly accept/reject)")
            print("    2. Use only switch regressor (not both switch and repeat)")
            print("    3. Orthogonalize switch/repeat w.r.t. other regressors")
        else:
            print("\n⚠ Switch and/or repeat regressors show moderate VIF.")
            print(f"    Switch VIF: {switch_vif:.2f}")
            print(f"    Repeat VIF: {repeat_vif:.2f}")
            print("  Proceed with caution, but generally acceptable.")
    
    return summary


def plot_vif_results(vif_df, output_dir):
    """
    Create visualizations of VIF results.
    """
    print("\nCreating VIF visualizations...")
    
    # Set style
    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    # 1. VIF by regressor (box plot)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Box plot
    ax1 = axes[0]
    order = vif_df.groupby('Regressor')['VIF'].median().sort_values(ascending=False).index
    sns.boxplot(data=vif_df, x='Regressor', y='VIF', order=order, ax=ax1)
    ax1.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (Moderate)')
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (High)')
    ax1.set_xlabel('Regressor', fontsize=14)
    ax1.set_ylabel('VIF', fontsize=14)
    ax1.set_title('VIF Distribution Across All Subjects/Runs', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bar plot of mean VIF
    ax2 = axes[1]
    mean_vif = vif_df.groupby('Regressor')['VIF'].mean().sort_values(ascending=False)
    colors = ['red' if v > 10 else ('orange' if v > 5 else 'green') for v in mean_vif.values]
    mean_vif.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
    ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Regressor', fontsize=14)
    ax2.set_ylabel('Mean VIF', fontsize=14)
    ax2.set_title('Mean VIF by Regressor', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'vif_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to: {plot_path}")
    plt.close()
    
    # 2. Heatmap of VIF by subject/run
    print("  Creating VIF heatmap...")
    
    # Pivot table
    pivot_data = vif_df.pivot_table(
        values='VIF',
        index='Regressor',
        columns=['Subject', 'Run'],
        aggfunc='first'
    )
    
    if len(pivot_data) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(pivot_data, annot=False, cmap='RdYlGn_r', 
                   vmin=0, vmax=15, cbar_kws={'label': 'VIF'},
                   ax=ax)
        ax.set_title('VIF Heatmap: Regressors × Subjects/Runs', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Subject_Run', fontsize=14)
        ax.set_ylabel('Regressor', fontsize=14)
        
        plt.tight_layout()
        
        heatmap_path = os.path.join(output_dir, 'vif_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved heatmap to: {heatmap_path}")
        plt.close()


def main():
    """
    Main function to compute VIF.
    """
    # ============================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ============================================================================
    
    # Paths
    BIDS_DIR = "/oak/stanford/groups/russpold/data/openneuro_fitlins/input/ds001734"
    DERIVATIVES_DIR = "/oak/stanford/groups/russpold/data/openneuro_fitlins/fmriprep/ds001734"
    
    # Output directory
    OUTPUT_DIR = "vif_analysis_results"
    
    # Subjects to process (update this list)
    if sys.argv[1] == "equalIndifference":
        SUBJECTS = [
            "001",
            "003",
            "005",
            "009",
            "011",
            "015",
            "017",
            "019",
            "021",
            "025",
            "027",
            "029",
            "033",
            "035",
            "037",
            "039",
            "041",
            "043",
            "045",
            "047",
            "049",
            "051",
            "053",
            "055",
            "057",
            "059",
            "061",
            "063",
            "067",
            "069",
            "071",
            "073",
            "075",
            "077",
            "079",
            "081",
            "083",
            "085",
            "087",
            "089",
            "093",
            "095",
            "099",
            "103",
            "105",
            "107",
            "109",
            "113",
            "115",
            "117",
            "119",
            "121",
            "123",
        ]
    elif sys.argv[1] == "equalRange":
        SUBJECTS = [
            "002",
            "004",
            "006",
            "008",
            "010",
            "014",
            "016",
            "018",
            "020",
            "022",
            "024",
            "026",
            "030",
            "032",
            "036",
            "038",
            "040",
            "044",
            "046",
            "050",
            "052",
            "054",
            "056",
            "058",
            "060",
            "062",
            "064",
            "066",
            "068",
            "070",
            "072",
            "074",
            "076",
            "080",
            "082",
            "084",
            "088",
            "090",
            "092",
            "094",
            "096",
            "098",
            "100",
            "102",
            "104",
            "106",
            "108",
            "110",
            "112",
            "114",
            "116",
            "118",
            "120",
            "124"
        ]
    
    # Runs
    RUNS = ['01', '02', '03', '04']
    
    # Regressors to check
    REGRESSORS = [
        'strongly_reject',
        'weakly_reject',
        'weakly_accept',
        'strongly_accept',
        'pos_sv',
        'neg_sv',
        'entropy',
        'RT',
        'switch',
        'repeat'
    ]
    
    # TR
    TR = 1.0
    
    # ============================================================================
    
    # Run analysis
    vif_results = analyze_all_subjects(
        bids_dir=BIDS_DIR,
        derivatives_dir=DERIVATIVES_DIR,
        subjects=SUBJECTS,
        runs=RUNS,
        regressors_of_interest=REGRESSORS,
        output_dir=OUTPUT_DIR,
        tr=TR
    )
    
    print("\n" + "="*60)
    print("VIF ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • vif_all_subjects_runs.csv - Full VIF results")
    print("  • vif_summary.csv - Summary statistics")
    print("  • vif_distribution.png - VIF visualizations")
    print("  • vif_heatmap.png - VIF by subject/run")
    print("\n" + "="*60)
    
    return vif_results


if __name__ == "__main__":
    results = main()
