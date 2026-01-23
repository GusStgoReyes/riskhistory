"""
Quick Analysis: Switch vs Repeat Activation
===========================================

Simple script to identify which brain regions are activated
in your switch analysis.

Usage:
1. Update the paths in the CONFIGURATION section
2. Run: python quick_switch_analysis.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
from nilearn.reporting import get_clusters_table
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt


def quick_analysis(stat_map_path, 
                  contrast_name,
                  threshold=3.1,
                  cluster_size=10,
                  output_dir='switch_results'):
    """
    Quick analysis of a single contrast.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {contrast_name}")
    print(f"{'='*60}\n")
    
    # Load the statistical map
    stat_img = nib.load(stat_map_path)
    stat_data = stat_img.get_fdata()
    
    print(f"Statistical map info:")
    print(f"  Min: {np.nanmin(stat_data):.2f}")
    print(f"  Max: {np.nanmax(stat_data):.2f}")
    print(f"  Voxels > {threshold}: {np.sum(stat_data > threshold)}")
    print(f"  Voxels < -{threshold}: {np.sum(stat_data < -threshold)}")
    
    # Threshold the map
    print(f"\nApplying threshold (t/z > {threshold}, cluster > {cluster_size} voxels)...")
    
    try:
        # Try FDR correction first
        thresholded_map, actual_threshold = threshold_stats_img(
            stat_img,
            alpha=0.05,
            height_control='fdr',
            cluster_threshold=cluster_size,
            two_sided=True
        )
        print(f"FDR-corrected threshold: {actual_threshold:.3f}")
    except:
        # Fall back to uncorrected
        print(f"Using uncorrected threshold: {threshold}")
        thresholded_data = stat_data.copy()
        thresholded_data[np.abs(thresholded_data) < threshold] = 0
        thresholded_map = nib.Nifti1Image(thresholded_data, stat_img.affine)
        actual_threshold = threshold
    
    # Get clusters
    try:
        clusters = get_clusters_table(
            thresholded_map,
            stat_threshold=actual_threshold,
            cluster_threshold=cluster_size,
            two_sided=True
        )
        
        if len(clusters) > 0:
            print(f"\nFound {len(clusters)} significant clusters!\n")
            
            # Simplify cluster table for readability
            if 'Cluster ID' in clusters.columns:
                summary = clusters[['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)']].copy()
            else:
                summary = clusters
            
            print(summary.to_string(index=False))
            
            # Save results
            csv_path = os.path.join(output_dir, f'{contrast_name}_clusters.csv')
            clusters.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")
        else:
            print("No significant clusters found at this threshold.")
            print("Try lowering the threshold or cluster size.")
            clusters = None
    except Exception as e:
        print(f"Could not extract clusters: {e}")
        clusters = None
    
    # Create visualization
    print("\nCreating visualization...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Glass brain
    ax1 = plt.subplot(2, 1, 1)
    plotting.plot_glass_brain(
        thresholded_map,
        display_mode='lyrz',
        colorbar=True,
        threshold=actual_threshold,
        title=f'{contrast_name} (Glass Brain)',
        axes=ax1
    )
    
    # Slice view
    ax2 = plt.subplot(2, 1, 2)
    plotting.plot_stat_map(
        thresholded_map,
        display_mode='z',
        cut_coords=8,
        threshold=actual_threshold,
        colorbar=True,
        title=f'{contrast_name} (Axial Slices)',
        axes=ax2
    )
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f'{contrast_name}_activation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")
    plt.close()
    
    return clusters


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

# UPDATE THESE PATHS TO YOUR ACTUAL FITLINS OUTPUT FILES

# Path to Switch > Repeat statistical map
SWITCH_STAT_MAP = "/scratch/users/gustxsr/openneuro_fitlins/analyses/ds001734/task-MGTM2equalIndifference/node-dataLevel/contrast-switchGreaterRepeat_stat-t_statmap.nii.gz"

# Path to Button Presses statistical map  
BUTTON_STAT_MAP = "/scratch/users/gustxsr/openneuro_fitlins/analyses/ds001734/task-MGTM2equalIndifference/node-dataLevel/contrast-buttonPresses_stat-t_statmap.nii.gz"

# Output directory
OUTPUT_DIR = "/scratch/users/gustxsr/switch_analysis_results"

# Threshold settings
THRESHOLD = 3.1  # t-value threshold (≈ p < 0.001 uncorrected)
CLUSTER_SIZE = 10  # Minimum cluster size in voxels

# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("SWITCH ANALYSIS: BRAIN ACTIVATION")
    print("="*60)
    
    # Analyze Switch > Repeat
    print("\n" + "="*60)
    print("1. SWITCH > REPEAT")
    print("="*60)
    switch_clusters = quick_analysis(
        SWITCH_STAT_MAP,
        'Switch_greater_Repeat',
        threshold=THRESHOLD,
        cluster_size=CLUSTER_SIZE,
        output_dir=OUTPUT_DIR
    )
    
    # Analyze Button Presses
    print("\n" + "="*60)
    print("2. BUTTON PRESSES")
    print("="*60)
    button_clusters = quick_analysis(
        BUTTON_STAT_MAP,
        'Button_Presses',
        threshold=THRESHOLD,
        cluster_size=CLUSTER_SIZE,
        output_dir=OUTPUT_DIR
    )
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nFiles created:")
    print("  - Switch_greater_Repeat_clusters.csv")
    print("  - Switch_greater_Repeat_activation.png")
    print("  - Button_Presses_clusters.csv")
    print("  - Button_Presses_activation.png")
    
    # Expected regions for Switch > Repeat
    print("\n" + "="*60)
    print("EXPECTED REGIONS (based on literature):")
    print("="*60)
    print("\nSwitch > Repeat typically shows:")
    print("  - Dorsolateral prefrontal cortex (dlPFC)")
    print("  - Anterior cingulate cortex (ACC)")
    print("  - Inferior parietal lobule")
    print("  - Pre-supplementary motor area (pre-SMA)")
    print("\nButton Presses typically shows:")
    print("  - Motor cortex")
    print("  - Supplementary motor area (SMA)")
    print("  - Visual cortex")
    print("  - Ventromedial prefrontal cortex (vmPFC)")
    print("\n" + "="*60)