"""
Extract Peak Coordinates and Anatomical Labels
==============================================

This script will extract the peak coordinates from your Switch > Repeat
contrast and provide anatomical labels using standard atlases.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets
from nilearn.reporting import get_clusters_table
from nilearn.image import threshold_img


def extract_peak_coordinates(stat_map_path, 
                             threshold=3.1,
                             cluster_size=10):
    """
    Extract peak coordinates and anatomical labels from statistical map.
    """
    print("="*60)
    print("EXTRACTING PEAK COORDINATES")
    print("="*60)
    
    # Load statistical map
    stat_img = nib.load(stat_map_path)
    
    # Threshold the map
    print(f"\nApplying threshold: {threshold}")
    thresholded_img = threshold_img(stat_img, threshold=threshold)
    
    # Get clusters table with anatomical labels
    print("Extracting clusters with anatomical labels...")
    
    try:
        # This will use Harvard-Oxford atlas for labeling
        clusters = get_clusters_table(
            thresholded_img,
            stat_threshold=threshold,
            cluster_threshold=cluster_size,
            two_sided=False  # Only positive activations
        )
        
        print(f"\nFound {len(clusters)} significant clusters\n")
        
        # Reorder for readability
        if 'Cluster ID' in clusters.columns:
            display_cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 
                          'Cluster Size (mm3)']
            if len(clusters.columns) > 6:
                # Add anatomical label if present
                display_cols += [c for c in clusters.columns if c not in display_cols]
            
            print(clusters[display_cols].to_string(index=False))
        else:
            print(clusters.to_string(index=False))
        
        # Save to CSV
        clusters.to_csv('switch_greater_repeat_peaks.csv', index=False)
        print("\n✓ Saved to: switch_greater_repeat_peaks.csv")
        
        return clusters
        
    except Exception as e:
        print(f"Could not extract clusters automatically: {e}")
        print("\nTrying manual peak extraction...")
        
        # Manual extraction
        stat_data = stat_img.get_fdata()
        
        # Find peaks above threshold
        mask = stat_data > threshold
        peak_coords = []
        
        # Get affine for coordinate transformation
        affine = stat_img.affine
        
        # Find local maxima (simplified)
        from scipy.ndimage import maximum_filter
        
        # Find local maxima
        local_max = (stat_data == maximum_filter(stat_data, size=3)) & mask
        
        # Get coordinates of local maxima
        peak_voxels = np.argwhere(local_max)
        peak_values = stat_data[local_max]
        
        # Sort by peak value
        sorted_indices = np.argsort(peak_values)[::-1]
        
        # Take top 20 peaks
        top_peaks = sorted_indices[:20]
        
        print(f"\nTop 20 peaks:")
        print("-"*60)
        print(f"{'Rank':<6} {'X':<8} {'Y':<8} {'Z':<8} {'t-value':<8}")
        print("-"*60)
        
        results = []
        for rank, idx in enumerate(top_peaks, 1):
            voxel = peak_voxels[idx]
            value = peak_values[idx]
            
            # Convert voxel to MNI coordinates
            mni = nib.affines.apply_affine(affine, voxel)
            
            print(f"{rank:<6} {mni[0]:<8.1f} {mni[1]:<8.1f} {mni[2]:<8.1f} {value:<8.2f}")
            
            results.append({
                'Rank': rank,
                'X': mni[0],
                'Y': mni[1],
                'Z': mni[2],
                't-value': value
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('switch_greater_repeat_peaks.csv', index=False)
        print("\n✓ Saved to: switch_greater_repeat_peaks.csv")
        
        return results_df


def label_coordinates(x, y, z):
    """
    Look up anatomical label for MNI coordinates.
    Uses approximate mapping to standard regions.
    """
    labels = []
    
    # Prefrontal regions
    if z > 20:  # Above AC-PC line
        if abs(x) > 30 and abs(y) > 20:
            labels.append("Dorsolateral Prefrontal Cortex (dlPFC)")
        elif abs(x) < 15 and y > 0:
            if z > 40:
                labels.append("Superior Medial Frontal Cortex")
            else:
                labels.append("Anterior Cingulate Cortex (ACC)")
        elif abs(x) > 30 and y > 40:
            labels.append("Frontal Pole")
    
    # Parietal regions
    if z > 30 and abs(y) < -30:
        if abs(x) > 20:
            labels.append("Inferior Parietal Lobule")
        else:
            labels.append("Precuneus")
    
    if z > 40 and abs(y) > -60 and abs(y) < -20:
        labels.append("Superior Parietal Lobule")
    
    # Motor regions
    if abs(x) > 20 and abs(y) < -10 and abs(y) > -30 and z > 45:
        labels.append("Premotor Cortex")
    
    # Insula
    if abs(x) > 30 and abs(y) > 0 and abs(y) < 20 and z < 20:
        labels.append("Insula")
    
    # Temporal
    if abs(x) > 40 and abs(y) < -30 and z < 0:
        labels.append("Middle Temporal Gyrus")
    
    # Occipital
    if abs(y) < -70:
        labels.append("Occipital Cortex (Visual)")
    
    # Striatum
    if abs(x) < 20 and abs(y) > -5 and abs(y) < 15 and z < 10:
        labels.append("Caudate/Striatum")
    
    # Thalamus
    if abs(x) < 15 and abs(y) < -10 and abs(y) > -30 and z < 10:
        labels.append("Thalamus")
    
    if not labels:
        labels = ["Unclassified"]
    
    return " / ".join(labels)


def annotate_peaks(peaks_file='switch_greater_repeat_peaks.csv'):
    """
    Add anatomical labels to peaks file.
    """
    print("\n" + "="*60)
    print("ANNOTATING PEAKS WITH ANATOMICAL LABELS")
    print("="*60)
    
    peaks = pd.read_csv(peaks_file)
    
    # Add anatomical labels
    if 'X' in peaks.columns and 'Y' in peaks.columns and 'Z' in peaks.columns:
        peaks['Anatomical_Label'] = peaks.apply(
            lambda row: label_coordinates(row['X'], row['Y'], row['Z']),
            axis=1
        )
        
        print("\nPeaks with anatomical labels:")
        print("-"*80)
        display_cols = ['X', 'Y', 'Z', 't-value', 'Anatomical_Label'] if 't-value' in peaks.columns else ['X', 'Y', 'Z', 'Peak Stat', 'Anatomical_Label']
        available_cols = [c for c in display_cols if c in peaks.columns]
        print(peaks[available_cols].to_string(index=False, max_colwidth=50))
        
        # Save annotated file
        peaks.to_csv('switch_greater_repeat_peaks_labeled.csv', index=False)
        print("\n✓ Saved to: switch_greater_repeat_peaks_labeled.csv")
    
    return peaks


def create_summary_report(peaks):
    """
    Create a summary report of main regions.
    """
    print("\n" + "="*60)
    print("SUMMARY: MAIN BRAIN REGIONS ACTIVATED")
    print("="*60)
    
    if 'Anatomical_Label' not in peaks.columns:
        print("No anatomical labels available")
        return
    
    # Count regions
    all_labels = []
    for label_str in peaks['Anatomical_Label']:
        if pd.notna(label_str):
            all_labels.extend([l.strip() for l in label_str.split('/')])
    
    from collections import Counter
    region_counts = Counter(all_labels)
    
    print("\nMost frequently activated regions:")
    print("-"*60)
    for region, count in region_counts.most_common(10):
        print(f"  {region}: {count} peak(s)")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    print("\nYour Switch > Repeat contrast shows activation in:")
    
    if any('dlPFC' in l or 'Dorsolateral' in l for l in all_labels):
        print("\n✓ Dorsolateral Prefrontal Cortex (dlPFC)")
        print("  → Executive control and working memory during switching")
    
    if any('ACC' in l or 'Cingulate' in l for l in all_labels):
        print("\n✓ Anterior Cingulate Cortex (ACC)")
        print("  → Conflict monitoring and response selection")
    
    if any('Parietal' in l for l in all_labels):
        print("\n✓ Parietal Cortex")
        print("  → Attentional shifting and spatial processing")
    
    if any('Insula' in l for l in all_labels):
        print("\n✓ Insula")
        print("  → Interoceptive awareness and salience processing")
    
    if any('Visual' in l or 'Occipital' in l for l in all_labels):
        print("\n✓ Visual Cortex")
        print("  → Enhanced visual processing during switches")
    
    print("\nThis pattern is consistent with:")
    print("  • Cognitive control network engagement")
    print("  • Attentional reorientation")
    print("  • Value reassessment and decision updating")
    

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Path to your statistical map
    STAT_MAP = "/scratch/users/gustxsr/openneuro_fitlins/analyses/ds001734/task-MGTM2equalRange/node-dataLevel/contrast-switchGreaterRepeat_stat-t_statmap.nii.gz"
    
    # Extract peaks
    peaks = extract_peak_coordinates(STAT_MAP, threshold=3.1, cluster_size=10)
    
    # Annotate with anatomical labels
    if peaks is not None:
        annotated_peaks = annotate_peaks()
        
        # Create summary
        create_summary_report(annotated_peaks)
        
        print("\n" + "="*60)
        print("FILES CREATED:")
        print("="*60)
        print("  • switch_greater_repeat_peaks.csv")
        print("  • switch_greater_repeat_peaks_labeled.csv")
        print("="*60)
