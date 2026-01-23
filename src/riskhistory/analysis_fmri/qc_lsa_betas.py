"""
Quality Control for LSA Trial Betas
===================================

This script performs quality control checks on LSA trial beta maps
and metadata before running MVPA analysis.

Author: Generated for NARPS mixed gambles analysis
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, image
import warnings
warnings.filterwarnings('ignore')


def check_metadata(metadata_path):
    """
    Check metadata file for completeness and issues.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata CSV
        
    Returns
    -------
    dict
        QC report
    """
    print("="*60)
    print("METADATA QUALITY CHECK")
    print("="*60)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    report = {}
    
    # Basic counts
    report['total_trials'] = len(metadata)
    report['n_subjects'] = metadata['subject'].nunique()
    report['n_runs_per_subject'] = metadata.groupby('subject')['run'].nunique().to_dict()
    
    print(f"\nTotal trials: {report['total_trials']}")
    print(f"Number of subjects: {report['n_subjects']}")
    print(f"Runs per subject: {metadata.groupby('subject')['run'].nunique().value_counts().to_dict()}")
    
    # Check for missing values
    print("\nMissing values:")
    missing = metadata.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({100*count/len(metadata):.1f}%)")
        report[f'missing_{col}'] = count
    
    # Check file existence
    print("\nChecking beta file existence...")
    if 'beta_path' in metadata.columns:
        existing_files = metadata['beta_path'].apply(os.path.exists)
        n_missing_files = (~existing_files).sum()
        print(f"  Missing beta files: {n_missing_files}/{len(metadata)}")
        report['missing_beta_files'] = n_missing_files
        
        if n_missing_files > 0:
            print("  WARNING: Some beta files are missing!")
            print("  First few missing:")
            print(metadata[~existing_files]['beta_path'].head())
    
    # Check label distribution
    if 'prev_choice_n1_binary' in metadata.columns:
        print("\nPrevious choice distribution (binary):")
        dist = metadata['prev_choice_n1_binary'].value_counts(dropna=False)
        print(dist)
        report['label_distribution_binary'] = dist.to_dict()
        
        # Check balance
        if len(dist[dist.index.notna()]) == 2:
            balance_ratio = dist.min() / dist.max()
            print(f"  Class balance ratio: {balance_ratio:.2f}")
            report['class_balance'] = balance_ratio
            
            if balance_ratio < 0.5:
                print("  WARNING: Classes are imbalanced!")
    
    # Check trials per subject
    print("\nTrials per subject:")
    trials_per_subject = metadata.groupby('subject').size()
    print(f"  Mean: {trials_per_subject.mean():.1f}")
    print(f"  Std: {trials_per_subject.std():.1f}")
    print(f"  Min: {trials_per_subject.min()}")
    print(f"  Max: {trials_per_subject.max()}")
    report['trials_per_subject'] = trials_per_subject.to_dict()
    
    # Check trials with valid labels per subject
    if 'prev_choice_n1_binary' in metadata.columns:
        print("\nTrials with valid previous choice labels per subject:")
        valid_per_subject = metadata[metadata['prev_choice_n1_binary'].notna()].groupby('subject').size()
        print(f"  Mean: {valid_per_subject.mean():.1f}")
        print(f"  Min: {valid_per_subject.min()}")
        print(f"  Max: {valid_per_subject.max()}")
        report['valid_trials_per_subject'] = valid_per_subject.to_dict()
        
        if valid_per_subject.min() < 15:
            print("  WARNING: Some subjects have very few valid trials for classification!")
    
    return report


def check_beta_maps(metadata_path, n_subjects=3, n_trials_per_subject=5):
    """
    Load and check beta maps for anomalies.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata CSV
    n_subjects : int
        Number of subjects to check
    n_trials_per_subject : int
        Number of trials to check per subject
        
    Returns
    -------
    dict
        QC report
    """
    print("\n" + "="*60)
    print("BETA MAP QUALITY CHECK")
    print("="*60)
    
    metadata = pd.read_csv(metadata_path)
    
    report = {
        'beta_means': [],
        'beta_stds': [],
        'beta_max': [],
        'beta_min': [],
        'n_zero_voxels': [],
        'n_nan_voxels': []
    }
    
    # Sample subjects
    subjects = metadata['subject'].unique()[:n_subjects]
    
    for subject in subjects:
        print(f"\nChecking subject {subject}...")
        subject_data = metadata[metadata['subject'] == subject]
        
        # Sample trials
        sample_trials = subject_data.sample(min(n_trials_per_subject, len(subject_data)))
        
        for _, trial in sample_trials.iterrows():
            beta_path = trial['beta_path']
            
            if not os.path.exists(beta_path):
                print(f"  WARNING: File not found: {beta_path}")
                continue
            
            # Load beta map
            beta_img = nib.load(beta_path)
            beta_data = beta_img.get_fdata()
            
            # Compute statistics
            report['beta_means'].append(np.mean(beta_data))
            report['beta_stds'].append(np.std(beta_data))
            report['beta_max'].append(np.max(beta_data))
            report['beta_min'].append(np.min(beta_data))
            report['n_zero_voxels'].append(np.sum(beta_data == 0))
            report['n_nan_voxels'].append(np.sum(np.isnan(beta_data)))
            
            print(f"  Trial {trial['trial_label']}:")
            print(f"    Mean: {np.mean(beta_data):.4f}")
            print(f"    Std: {np.std(beta_data):.4f}")
            print(f"    Range: [{np.min(beta_data):.4f}, {np.max(beta_data):.4f}]")
            print(f"    Zero voxels: {np.sum(beta_data == 0)}")
            print(f"    NaN voxels: {np.sum(np.isnan(beta_data))}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("BETA MAP SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nMean beta across trials:")
    print(f"  Mean: {np.mean(report['beta_means']):.4f}")
    print(f"  Std: {np.std(report['beta_means']):.4f}")
    
    print(f"\nStd beta across trials:")
    print(f"  Mean: {np.mean(report['beta_stds']):.4f}")
    print(f"  Std: {np.std(report['beta_stds']):.4f}")
    
    # Check for anomalies
    print("\n" + "="*60)
    print("ANOMALY DETECTION")
    print("="*60)
    
    # Check for NaN values
    if np.any(np.array(report['n_nan_voxels']) > 0):
        print("  WARNING: Some beta maps contain NaN values!")
    
    # Check for extreme values
    if np.max(np.abs(report['beta_max'])) > 100:
        print("  WARNING: Some beta maps have very large values (> 100)!")
    
    # Check for all-zero maps
    n_voxels = np.prod(beta_img.shape)
    if np.any(np.array(report['n_zero_voxels']) > 0.9 * n_voxels):
        print("  WARNING: Some beta maps are mostly zero!")
    
    return report


def plot_qc_figures(metadata_path, output_dir='qc_figures'):
    """
    Generate QC plots.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata CSV
    output_dir : str
        Directory to save figures
    """
    print("\n" + "="*60)
    print("GENERATING QC FIGURES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = pd.read_csv(metadata_path)
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. Trials per subject
    plt.figure(figsize=(12, 6))
    trials_per_subject = metadata.groupby('subject').size()
    plt.subplot(1, 2, 1)
    trials_per_subject.plot(kind='bar')
    plt.xlabel('Subject')
    plt.ylabel('Number of trials')
    plt.title('Total trials per subject')
    plt.xticks(rotation=45)
    
    # 2. Valid trials per subject (for MVPA)
    if 'prev_choice_n1_binary' in metadata.columns:
        plt.subplot(1, 2, 2)
        valid_trials = metadata[metadata['prev_choice_n1_binary'].notna()].groupby('subject').size()
        valid_trials.plot(kind='bar', color='green')
        plt.xlabel('Subject')
        plt.ylabel('Number of valid trials')
        plt.title('Trials with previous choice label')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trials_per_subject.png'), dpi=300)
    print(f"  Saved: {os.path.join(output_dir, 'trials_per_subject.png')}")
    plt.close()
    
    # 3. Class distribution
    if 'prev_choice_n1_binary' in metadata.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall distribution
        axes[0, 0].pie(metadata['prev_choice_n1_binary'].value_counts(dropna=True),
                      labels=metadata['prev_choice_n1_binary'].value_counts(dropna=True).index,
                      autopct='%1.1f%%')
        axes[0, 0].set_title('Overall previous choice distribution')
        
        # Distribution by subject
        subject_dist = metadata.groupby(['subject', 'prev_choice_n1_binary']).size().unstack(fill_value=0)
        subject_dist.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Subject')
        axes[0, 1].set_ylabel('Number of trials')
        axes[0, 1].set_title('Previous choice distribution by subject')
        axes[0, 1].legend(title='Previous choice')
        
        # Switch rate by subject
        if 'is_switch' in metadata.columns:
            switch_rate = metadata.groupby('subject')['is_switch'].mean()
            axes[1, 0].bar(range(len(switch_rate)), switch_rate.values)
            axes[1, 0].set_xlabel('Subject')
            axes[1, 0].set_ylabel('Switch rate')
            axes[1, 0].set_title('Switch rate by subject')
            axes[1, 0].axhline(0.5, color='red', linestyle='--', label='Chance')
            axes[1, 0].legend()
        
        # Response time distribution
        if 'RT' in metadata.columns:
            metadata['RT'].hist(bins=30, ax=axes[1, 1])
            axes[1, 1].set_xlabel('RT (seconds)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Response time distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distributions.png'), dpi=300)
        print(f"  Saved: {os.path.join(output_dir, 'class_distributions.png')}")
        plt.close()
    
    # 4. Sample beta maps visualization
    print("\n  Generating sample beta map visualizations...")
    
    # Get one trial from first subject
    first_trial = metadata.iloc[0]
    if os.path.exists(first_trial['beta_path']):
        beta_img = nib.load(first_trial['beta_path'])
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Stat map
        plotting.plot_stat_map(
            beta_img,
            title=f"Sample beta map: {first_trial['trial_label']}",
            display_mode='z',
            cut_coords=7,
            axes=axes[0]
        )
        
        # Glass brain
        plotting.plot_glass_brain(
            beta_img,
            title=f"Sample beta map (glass brain): {first_trial['trial_label']}",
            display_mode='lyrz',
            axes=axes[1]
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_beta_map.png'), dpi=300)
        print(f"  Saved: {os.path.join(output_dir, 'sample_beta_map.png')}")
        plt.close()
    
    print(f"\nAll QC figures saved to {output_dir}")


def generate_qc_report(metadata_path, output_path='qc_report.txt'):
    """
    Generate comprehensive QC report.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata CSV
    output_path : str
        Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LSA TRIAL BETA EXTRACTION - QUALITY CONTROL REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Metadata check
        f.write("1. METADATA CHECK\n")
        f.write("-"*60 + "\n")
        report1 = check_metadata(metadata_path)
        
        # Beta map check
        f.write("\n2. BETA MAP CHECK\n")
        f.write("-"*60 + "\n")
        report2 = check_beta_maps(metadata_path)
        
        f.write("\n" + "="*60 + "\n")
        f.write("QC REPORT COMPLETE\n")
        f.write("="*60 + "\n")
    
    print(f"\nFull QC report saved to {output_path}")


def main():
    """
    Main QC function.
    """
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    METADATA_PATH = "/scratch/users/gustxsr/lsa_trial_betas/all_subjects_trial_metadata.csv"
    OUTPUT_DIR = "/scratch/users/gustxsr/lsa_trial_betas/qc"
    
    # ============================================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("LSA TRIAL BETA QUALITY CONTROL")
    print("="*60)
    
    # Check metadata
    metadata_report = check_metadata(METADATA_PATH)
    
    # Check beta maps
    beta_report = check_beta_maps(METADATA_PATH, n_subjects=3, n_trials_per_subject=5)
    
    # Generate plots
    plot_qc_figures(METADATA_PATH, output_dir=OUTPUT_DIR)
    
    # Generate text report
    # generate_qc_report(METADATA_PATH, 
    #                   output_path=os.path.join(OUTPUT_DIR, 'qc_report.txt'))
    
    print("\n" + "="*60)
    print("QUALITY CONTROL COMPLETE")
    print("="*60)
    print(f"\nQC outputs saved to: {OUTPUT_DIR}")
    
    return metadata_report, beta_report


if __name__ == "__main__":
    metadata_report, beta_report = main()