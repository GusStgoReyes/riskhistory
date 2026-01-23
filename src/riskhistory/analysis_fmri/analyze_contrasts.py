"""
Identify Significant Brain Regions from GLM Contrasts
=====================================================

This script identifies which brain regions are significantly activated
in your switch analysis contrasts.

Author: Generated for NARPS mixed gambles analysis
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, image, datasets
from nilearn.reporting import get_clusters_table
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def identify_significant_regions(stat_map_path,
                                 contrast_name,
                                 output_dir='results',
                                 threshold=3.1,  # Corresponds to p < 0.001 uncorrected
                                 cluster_threshold=10,  # Minimum cluster size in voxels
                                 two_sided=True,
                                 height_control='fdr',  # 'fdr', 'bonferroni', 'fpr', or None
                                 alpha=0.05):
    """
    Identify significant brain regions from a statistical map.
    
    Parameters
    ----------
    stat_map_path : str
        Path to statistical map (z-stat or t-stat)
    contrast_name : str
        Name of the contrast for labeling
    output_dir : str
        Directory to save results
    threshold : float
        Statistical threshold (z-score or t-score)
    cluster_threshold : int
        Minimum cluster size in voxels
    two_sided : bool
        Whether to look at both positive and negative activations
    height_control : str or None
        Multiple comparison correction method
        Options: 'fdr', 'bonferroni', 'fpr', or None
    alpha : float
        Significance level for correction
        
    Returns
    -------
    pd.DataFrame
        Table of significant clusters with anatomical labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {contrast_name}")
    print(f"{'='*60}")
    
    # Load statistical map
    stat_img = nib.load(stat_map_path)
    print(f"\nLoaded stat map: {stat_map_path}")
    print(f"  Shape: {stat_img.shape}")
    
    # Get data statistics
    stat_data = stat_img.get_fdata()
    print(f"\nStat map statistics:")
    print(f"  Min: {np.nanmin(stat_data):.3f}")
    print(f"  Max: {np.nanmax(stat_data):.3f}")
    print(f"  Mean: {np.nanmean(stat_data):.3f}")
    print(f"  Std: {np.nanstd(stat_data):.3f}")
    
    # Apply thresholding
    print(f"\nApplying threshold:")
    print(f"  Method: {height_control if height_control else 'uncorrected'}")
    print(f"  Threshold: {threshold}")
    print(f"  Cluster size: {cluster_threshold} voxels")
    
    try:
        thresholded_map, threshold_value = threshold_stats_img(
            stat_img,
            alpha=alpha,
            height_control=height_control,
            cluster_threshold=cluster_threshold,
            two_sided=two_sided
        )
        print(f"  Actual threshold applied: {threshold_value:.3f}")
    except Exception as e:
        print(f"  Warning: Automatic thresholding failed ({e})")
        print(f"  Using manual threshold: {threshold}")
        # Manual thresholding
        thresholded_data = stat_data.copy()
        if two_sided:
            thresholded_data[np.abs(thresholded_data) < threshold] = 0
        else:
            thresholded_data[thresholded_data < threshold] = 0
        thresholded_map = nib.Nifti1Image(thresholded_data, stat_img.affine)
        threshold_value = threshold
    
    # Get clusters table
    print("\nExtracting significant clusters...")
    
    try:
        # Try to get clusters table with anatomical labels
        clusters_table = get_clusters_table(
            thresholded_map,
            stat_threshold=threshold_value,
            cluster_threshold=cluster_threshold,
            two_sided=two_sided
        )
        
        if len(clusters_table) == 0:
            print("  No significant clusters found at this threshold!")
            return None
        
        print(f"  Found {len(clusters_table)} significant clusters")
        
        # Add contrast name to table
        clusters_table['Contrast'] = contrast_name
        
        # Reorder columns for readability
        col_order = ['Contrast', 'Cluster ID', 'X', 'Y', 'Z', 
                    'Peak Stat', 'Cluster Size (mm3)']
        available_cols = [col for col in col_order if col in clusters_table.columns]
        other_cols = [col for col in clusters_table.columns if col not in available_cols]
        clusters_table = clusters_table[available_cols + other_cols]
        
        # Save table
        csv_path = os.path.join(output_dir, f'{contrast_name}_clusters.csv')
        clusters_table.to_csv(csv_path, index=False)
        print(f"\nSaved clusters table to: {csv_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SIGNIFICANT CLUSTERS")
        print("="*60)
        print(clusters_table.to_string(index=False, max_colwidth=30))
        
    except Exception as e:
        print(f"  Warning: Could not extract cluster table ({e})")
        print("  Creating basic cluster summary instead...")
        clusters_table = None
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # 1. Glass brain
    fig = plt.figure(figsize=(12, 4))
    
    display = plotting.plot_glass_brain(
        thresholded_map,
        display_mode='lyrz',
        colorbar=True,
        plot_abs=False,
        threshold=threshold_value,
        title=f'{contrast_name}\n(threshold: {threshold_value:.2f}, cluster size: {cluster_threshold} voxels)',
        figure=fig
    )
    
    glass_path = os.path.join(output_dir, f'{contrast_name}_glass_brain.png')
    plt.savefig(glass_path, dpi=300, bbox_inches='tight')
    print(f"  Saved glass brain to: {glass_path}")
    plt.close()
    
    # 2. Statistical map on slices
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Axial
    plotting.plot_stat_map(
        thresholded_map,
        display_mode='z',
        cut_coords=7,
        threshold=threshold_value,
        colorbar=True,
        title=f'{contrast_name} - Axial',
        axes=axes[0]
    )
    
    # Sagittal
    plotting.plot_stat_map(
        thresholded_map,
        display_mode='x',
        cut_coords=7,
        threshold=threshold_value,
        colorbar=True,
        title=f'{contrast_name} - Sagittal',
        axes=axes[1]
    )
    
    # Coronal
    plotting.plot_stat_map(
        thresholded_map,
        display_mode='y',
        cut_coords=7,
        threshold=threshold_value,
        colorbar=True,
        title=f'{contrast_name} - Coronal',
        axes=axes[2]
    )
    
    plt.tight_layout()
    slices_path = os.path.join(output_dir, f'{contrast_name}_slices.png')
    plt.savefig(slices_path, dpi=300, bbox_inches='tight')
    print(f"  Saved slice views to: {slices_path}")
    plt.close()
    
    # 3. Surface projection (if MNI space)
    try:
        fig = plt.figure(figsize=(12, 8))
        plotting.plot_surf_stat_map(
            datasets.fetch_surf_fsaverage()['infl_left'],
            thresholded_map,
            hemi='left',
            title=f'{contrast_name} - Left Hemisphere',
            threshold=threshold_value,
            bg_map=datasets.fetch_surf_fsaverage()['sulc_left'],
            figure=fig
        )
        surf_path = os.path.join(output_dir, f'{contrast_name}_surface.png')
        plt.savefig(surf_path, dpi=300, bbox_inches='tight')
        print(f"  Saved surface projection to: {surf_path}")
        plt.close()
    except Exception as e:
        print(f"  Could not create surface projection: {e}")
    
    print(f"\n{'='*60}\n")
    
    return clusters_table


def analyze_multiple_contrasts(contrast_paths, 
                               output_dir='results',
                               threshold=3.1,
                               cluster_threshold=10,
                               height_control='fdr',
                               alpha=0.05):
    """
    Analyze multiple contrasts and create a combined summary.
    
    Parameters
    ----------
    contrast_paths : dict
        Dictionary mapping contrast names to file paths
    output_dir : str
        Directory to save results
    threshold : float
        Statistical threshold
    cluster_threshold : int
        Minimum cluster size
    height_control : str or None
        Multiple comparison correction method
    alpha : float
        Significance level
        
    Returns
    -------
    pd.DataFrame
        Combined clusters table for all contrasts
    """
    all_clusters = []
    
    for contrast_name, stat_map_path in contrast_paths.items():
        clusters = identify_significant_regions(
            stat_map_path=stat_map_path,
            contrast_name=contrast_name,
            output_dir=output_dir,
            threshold=threshold,
            cluster_threshold=cluster_threshold,
            two_sided=True,
            height_control=height_control,
            alpha=alpha
        )
        
        if clusters is not None:
            all_clusters.append(clusters)
    
    if len(all_clusters) > 0:
        # Combine all clusters
        combined_clusters = pd.concat(all_clusters, ignore_index=True)
        
        # Save combined table
        combined_path = os.path.join(output_dir, 'all_contrasts_clusters.csv')
        combined_clusters.to_csv(combined_path, index=False)
        print(f"Saved combined clusters table to: {combined_path}")
        
        return combined_clusters
    else:
        print("No significant clusters found in any contrast")
        return None


def create_comparison_figure(contrast_paths, output_dir='results', threshold=3.1):
    """
    Create a figure comparing multiple contrasts side-by-side.
    
    Parameters
    ----------
    contrast_paths : dict
        Dictionary mapping contrast names to file paths
    output_dir : str
        Directory to save figure
    threshold : float
        Statistical threshold
    """
    n_contrasts = len(contrast_paths)
    fig, axes = plt.subplots(n_contrasts, 1, figsize=(14, 4*n_contrasts))
    
    if n_contrasts == 1:
        axes = [axes]
    
    for idx, (contrast_name, stat_map_path) in enumerate(contrast_paths.items()):
        stat_img = nib.load(stat_map_path)
        
        plotting.plot_stat_map(
            stat_img,
            display_mode='z',
            cut_coords=7,
            threshold=threshold,
            colorbar=True,
            title=contrast_name,
            axes=axes[idx]
        )
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'contrast_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved contrast comparison to: {comparison_path}")
    plt.close()


def main():
    """
    Main function to analyze GLM contrasts.
    """
    # ============================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ============================================================================
    
    # Base directory where fitlins saved results
    FITLINS_OUTPUT = "/path/to/fitlins/output"
    
    # Path template for statistical maps
    # Usually: {fitlins_output}/sub-{subject}/func/sub-{subject}_task-MGT_contrast-{contrast}_stat-effect_statmap.nii.gz
    
    # For group-level analysis, the path might be:
    # {fitlins_output}/dataset/func/contrast-{contrast}_stat-t_statmap.nii.gz

    # Define your contrasts
    CONTRAST_PATHS = {
        'Switch_greater_Repeat': "/scratch/users/gustxsr/openneuro_fitlins/analyses/ds001734/task-MGTM2equalRange/node-dataLevel/contrast-switchGreaterRepeat_stat-t_statmap.nii.gz",
        'Button_Presses': "/scratch/users/gustxsr/openneuro_fitlins/analyses/ds001734/task-MGTM2equalRange/node-dataLevel/contrast-buttonPresses_stat-t_statmap.nii.gz",
    }
    
    # Output directory for results
    OUTPUT_DIR = '/scratch/users/gustxsr/switch_analysis_results'
    
    # Statistical thresholding parameters
    THRESHOLD = 3.1  # z or t threshold (3.1 ≈ p < 0.001 uncorrected)
    CLUSTER_THRESHOLD = 10  # Minimum cluster size in voxels
    HEIGHT_CONTROL = 'fdr'  # 'fdr', 'bonferroni', or None for uncorrected
    ALPHA = 0.05  # Significance level for correction
    
    # ============================================================================
    
    print("="*60)
    print("GLM CONTRAST ANALYSIS: IDENTIFYING SIGNIFICANT REGIONS")
    print("="*60)
    
    # Analyze all contrasts
    combined_results = analyze_multiple_contrasts(
        contrast_paths=CONTRAST_PATHS,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
        cluster_threshold=CLUSTER_THRESHOLD,
        height_control=HEIGHT_CONTROL,
        alpha=ALPHA
    )
    
    # Create comparison figure
    create_comparison_figure(
        contrast_paths=CONTRAST_PATHS,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - *_clusters.csv: Cluster tables with coordinates")
    print("  - *_glass_brain.png: Glass brain visualizations")
    print("  - *_slices.png: Slice views")
    print("  - all_contrasts_clusters.csv: Combined results")
    print("  - contrast_comparison.png: Side-by-side comparison")
    
    return combined_results


if __name__ == "__main__":
    results = main()
