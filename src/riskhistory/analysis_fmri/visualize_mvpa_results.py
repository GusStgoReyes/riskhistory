"""
Visualize MVPA Classification Results
=====================================

This script creates publication-quality figures for MVPA classification results.

Author: Generated for NARPS mixed gambles analysis
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def plot_individual_accuracies(results_path, output_dir='figures', chance_level=0.5):
    """
    Plot individual subject classification accuracies.
    
    Parameters
    ----------
    results_path : str
        Path to group_classification_results.csv
    output_dir : str
        Directory to save figures
    chance_level : float
        Chance level for classification
    """
    # Load results
    results = pd.read_csv(results_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Plot 1: Bar plot with error bars =====
    ax1 = axes[0]
    
    # Sort by accuracy
    results_sorted = results.sort_values('cv_accuracy', ascending=False)
    
    # Create color map (significant in one color, non-significant in another)
    colors = ['#2ecc71' if sig else '#e74c3c' 
             for sig in results_sorted['significant']]
    
    # Bar plot
    x_pos = np.arange(len(results_sorted))
    bars = ax1.bar(x_pos, results_sorted['cv_accuracy'], 
                   yerr=results_sorted['cv_std'],
                   color=colors, alpha=0.7, capsize=5)
    
    # Chance line
    ax1.axhline(chance_level, color='black', linestyle='--', 
                linewidth=2, label=f'Chance ({chance_level:.0%})')
    
    # Formatting
    ax1.set_xlabel('Subject', fontsize=14)
    ax1.set_ylabel('Classification Accuracy', fontsize=14)
    ax1.set_title('Individual Subject Classification Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_sorted['subject'], rotation=45, ha='right')
    ax1.set_ylim([0.3, 1.0])
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Not significant')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # ===== Plot 2: Distribution and statistics =====
    ax2 = axes[1]
    
    # Violin plot
    parts = ax2.violinplot([results['cv_accuracy']], positions=[0.5], 
                          showmeans=True, showmedians=True, widths=0.7)
    
    # Customize violin
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    
    # Individual points
    y_jitter = np.random.normal(0.5, 0.02, len(results))
    ax2.scatter(y_jitter, results['cv_accuracy'], 
               alpha=0.6, s=100, color='navy', edgecolors='black', linewidth=1)
    
    # Chance line
    ax2.axhline(chance_level, color='black', linestyle='--', 
                linewidth=2, label=f'Chance ({chance_level:.0%})')
    
    # Mean line
    mean_acc = results['cv_accuracy'].mean()
    ax2.axhline(mean_acc, color='red', linestyle='-', 
                linewidth=2, label=f'Group Mean ({mean_acc:.1%})')
    
    # Add statistics text
    t_stat, p_val = stats.ttest_1samp(results['cv_accuracy'], chance_level)
    cohens_d = (mean_acc - chance_level) / results['cv_accuracy'].std()
    
    stats_text = f"Group Statistics:\n"
    stats_text += f"Mean ± SD: {mean_acc:.1%} ± {results['cv_accuracy'].std():.1%}\n"
    stats_text += f"Range: [{results['cv_accuracy'].min():.1%}, {results['cv_accuracy'].max():.1%}]\n"
    stats_text += f"t({len(results)-1}) = {t_stat:.2f}, p = {p_val:.4f}\n"
    stats_text += f"Cohen's d = {cohens_d:.2f}\n"
    stats_text += f"N significant: {results['significant'].sum()}/{len(results)}"
    
    ax2.text(0.95, 0.05, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    ax2.set_ylabel('Classification Accuracy', fontsize=14)
    ax2.set_title('Group Distribution', fontsize=16, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_ylim([0.3, 1.0])
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'individual_accuracies.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also save as PDF for publications
    output_path_pdf = os.path.join(output_dir, 'individual_accuracies.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")
    
    plt.close()


def plot_accuracy_by_trials(results_path, output_dir='figures'):
    """
    Plot relationship between number of trials and accuracy.
    
    Parameters
    ----------
    results_path : str
        Path to group_classification_results.csv
    output_dir : str
        Directory to save figures
    """
    results = pd.read_csv(results_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    colors = ['#2ecc71' if sig else '#e74c3c' for sig in results['significant']]
    plt.scatter(results['n_trials'], results['cv_accuracy'], 
               s=150, alpha=0.7, c=colors, edgecolors='black', linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(results['n_trials'], results['cv_accuracy'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results['n_trials'].min(), results['n_trials'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label='Trend')
    
    # Correlation
    corr, p_val = stats.pearsonr(results['n_trials'], results['cv_accuracy'])
    
    # Add correlation text
    corr_text = f"Pearson r = {corr:.3f}, p = {p_val:.4f}"
    plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    plt.xlabel('Number of Trials', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.title('Classification Accuracy vs. Number of Trials', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Not significant'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Trend')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'accuracy_vs_trials.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def create_summary_table(results_path, output_dir='figures'):
    """
    Create a summary table of results.
    
    Parameters
    ----------
    results_path : str
        Path to group_classification_results.csv
    output_dir : str
        Directory to save table
    """
    results = pd.read_csv(results_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary statistics
    summary_stats = {
        'Metric': [
            'Number of subjects',
            'Mean accuracy',
            'Std accuracy',
            'Min accuracy',
            'Max accuracy',
            'Median accuracy',
            'Subjects above chance (p < 0.05)',
            'Group t-statistic',
            'Group p-value',
            "Cohen's d"
        ],
        'Value': []
    }
    
    # Calculate statistics
    n_subjects = len(results)
    mean_acc = results['cv_accuracy'].mean()
    std_acc = results['cv_accuracy'].std()
    min_acc = results['cv_accuracy'].min()
    max_acc = results['cv_accuracy'].max()
    median_acc = results['cv_accuracy'].median()
    n_sig = results['significant'].sum()
    
    t_stat, p_val = stats.ttest_1samp(results['cv_accuracy'], 0.5)
    cohens_d = (mean_acc - 0.5) / std_acc
    
    # Fill in values
    summary_stats['Value'] = [
        f'{n_subjects}',
        f'{mean_acc:.3f} ({mean_acc*100:.1f}%)',
        f'{std_acc:.3f}',
        f'{min_acc:.3f} ({min_acc*100:.1f}%)',
        f'{max_acc:.3f} ({max_acc*100:.1f}%)',
        f'{median_acc:.3f} ({median_acc*100:.1f}%)',
        f'{n_sig}/{n_subjects} ({100*n_sig/n_subjects:.1f}%)',
        f'{t_stat:.3f}',
        f'{p_val:.6f}',
        f'{cohens_d:.3f}'
    ]
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('MVPA Classification: Summary Statistics', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Save
    output_path = os.path.join(output_dir, 'summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    plt.close()
    
    return summary_df


def plot_permutation_distributions(results_path, metadata_path, output_dir='figures'):
    """
    Plot permutation test distributions for a few example subjects.
    
    Note: This requires the full results with permutation scores saved.
    This is a placeholder for when you save detailed results.
    """
    # This would require saving the permutation scores from each subject
    # For now, we'll skip this but keep as a placeholder
    pass


def main():
    """
    Main function to generate all visualization.
    """
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    RESULTS_PATH = "/scratch/users/gustxsr/mvpa_results/group_classification_results.csv"
    OUTPUT_DIR = "/scratch/users/gustxsr/mvpa_results/figures"
    CHANCE_LEVEL = 0.5  # 50% for binary classification
    
    # ============================================================================
    
    print("="*60)
    print("GENERATING MVPA RESULT VISUALIZATIONS")
    print("="*60)
    
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found at {RESULTS_PATH}")
        print("Please run mvpa_classification.py first")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate plots
    print("\n1. Plotting individual accuracies...")
    plot_individual_accuracies(RESULTS_PATH, OUTPUT_DIR, CHANCE_LEVEL)
    
    print("\n2. Plotting accuracy vs. number of trials...")
    plot_accuracy_by_trials(RESULTS_PATH, OUTPUT_DIR)
    
    print("\n3. Creating summary table...")
    summary_df = create_summary_table(RESULTS_PATH, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - individual_accuracies.png/pdf")
    print("  - accuracy_vs_trials.png")
    print("  - summary_table.png")
    print("  - summary_statistics.csv")
    print("\n" + "="*60)
    
    # Print summary to console
    print("\nSUMMARY STATISTICS:")
    print(summary_df.to_string(index=False))
    print("\n" + "="*60)


if __name__ == "__main__":
    main()