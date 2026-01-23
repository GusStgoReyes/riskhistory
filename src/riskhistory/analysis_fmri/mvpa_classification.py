"""
MVPA Classification: Decoding Choice History
============================================

This script performs multivariate pattern analysis (MVPA) to decode
previous choices from current trial brain activity patterns.

Author: Generated for NARPS mixed gambles analysis
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, masking
from nilearn.decoding import SearchLight
from nilearn.image import load_img
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ChoiceHistoryMVPA:
    """
    MVPA classifier for decoding choice history from fMRI data.
    
    Parameters
    ----------
    labels_path : str
        Path to MVPA labels CSV file
    mask_img : str or Nifti1Image
        Brain mask for analysis
    output_dir : str
        Directory to save results
    """
    
    def __init__(self, labels_path, mask_img=None, output_dir='mvpa_results'):
        self.labels_df = pd.read_csv(labels_path)
        self.mask_img = mask_img
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loaded {len(self.labels_df)} trials")
        print(f"Subjects: {self.labels_df['subject'].nunique()}")
        
    def prepare_subject_data(self, subject, label_column='prev_choice_n1_binary'):
        """
        Prepare data for one subject.
        
        Parameters
        ----------
        subject : str
            Subject ID
        label_column : str
            Column name for classification labels
            
        Returns
        -------
        X : ndarray
            Brain data (trials × voxels)
        y : ndarray
            Labels
        groups : ndarray
            Run labels for cross-validation
        """
        # Get trials for this subject
        subject_data = self.labels_df[self.labels_df['subject'] == subject].copy()
        
        # Remove trials with missing labels
        subject_data = subject_data[subject_data[label_column].notna()]
        
        if len(subject_data) == 0:
            return None, None, None
        
        # Load beta maps
        beta_imgs = [load_img(path) for path in subject_data['beta_path']]
        
        # Concatenate into 4D image
        beta_4d = image.concat_imgs(beta_imgs)
        
        # Apply mask and get 2D array (trials × voxels)
        if self.mask_img is not None:
            X = masking.apply_mask(beta_4d, self.mask_img)
        else:
            # Create mask from data
            mask = image.math_img('np.mean(np.abs(img), axis=-1) > 0', img=beta_4d)
            X = masking.apply_mask(beta_4d, mask)
        
        # Get labels
        y = subject_data[label_column].values
        
        # Get run labels for cross-validation
        groups = subject_data['run'].values
        
        return X, y, groups
    
    def within_subject_classification(self, 
                                     subject,
                                     label_column='prev_choice_n1_binary',
                                     classifier=None,
                                     cv_strategy='leave_one_run_out',
                                     n_permutations=100):
        """
        Perform within-subject classification with cross-validation.
        
        Parameters
        ----------
        subject : str
            Subject ID
        label_column : str
            Column name for classification labels
        classifier : sklearn classifier or None
            Classifier to use (if None, uses linear SVM)
        cv_strategy : str
            Cross-validation strategy ('leave_one_run_out' or 'stratified')
        n_permutations : int
            Number of permutations for significance testing
            
        Returns
        -------
        dict
            Results dictionary with scores and statistics
        """
        print(f"\nClassifying subject {subject}...")
        
        # Prepare data
        X, y, groups = self.prepare_subject_data(subject, label_column)
        
        if X is None:
            print(f"No valid data for subject {subject}")
            return None
        
        print(f"  Data shape: {X.shape}")
        print(f"  Class distribution: {np.unique(y, return_counts=True)}")
        
        # Set up classifier
        if classifier is None:
            classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='linear', C=1.0))
            ])
        
        # Set up cross-validation
        if cv_strategy == 'leave_one_run_out':
            cv = LeaveOneGroupOut()
            cv_splits = list(cv.split(X, y, groups))
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(groups))))
            cv_splits = cv
        
        # Perform cross-validation
        print("  Running cross-validation...")
        scores = cross_val_score(classifier, X, y, cv=cv_splits, scoring='accuracy')
        
        print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Fold accuracies: {scores}")
        
        # Permutation test
        print(f"  Running permutation test ({n_permutations} permutations)...")
        score, perm_scores, pvalue = permutation_test_score(
            classifier, X, y, 
            cv=cv_splits,
            n_permutations=n_permutations,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        print(f"  Permutation test p-value: {pvalue:.4f}")
        
        # Compile results
        results = {
            'subject': subject,
            'n_trials': len(y),
            'n_voxels': X.shape[1],
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'cv_scores': scores,
            'permutation_score': score,
            'permutation_pvalue': pvalue,
            'permutation_scores': perm_scores,
            'chance_level': 1.0 / len(np.unique(y))
        }
        
        return results
    
    def searchlight_analysis(self,
                           subject,
                           label_column='prev_choice_n1_binary',
                           radius=5.0,
                           n_jobs=-1):
        """
        Perform searchlight analysis for one subject.
        
        Parameters
        ----------
        subject : str
            Subject ID
        label_column : str
            Column name for classification labels
        radius : float
            Searchlight radius in mm
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        Nifti1Image
            Searchlight accuracy map
        """
        print(f"\nSearchlight analysis for subject {subject}...")
        print(f"  Radius: {radius} mm")
        
        # Prepare data
        X, y, groups = self.prepare_subject_data(subject, label_column)
        
        if X is None:
            print(f"No valid data for subject {subject}")
            return None
        
        # Load one beta image to get affine and shape
        example_img = load_img(self.labels_df[self.labels_df['subject'] == subject].iloc[0]['beta_path'])
        
        # Set up searchlight
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0))
        ])
        
        cv = LeaveOneGroupOut()
        
        searchlight = SearchLight(
            mask_img=self.mask_img,
            radius=radius,
            estimator=classifier,
            cv=cv,
            n_jobs=n_jobs,
            scoring='accuracy',
            verbose=1
        )
        
        # Reconstruct 4D image for searchlight
        beta_imgs = [load_img(path) for path in 
                     self.labels_df[self.labels_df['subject'] == subject]['beta_path']]
        beta_4d = image.concat_imgs(beta_imgs)
        
        # Fit searchlight
        print("  Fitting searchlight (this may take a while)...")
        searchlight.fit(beta_4d, y, groups=groups)
        
        # Get accuracy map
        accuracy_map = searchlight.scores_
        
        # Save accuracy map
        output_path = os.path.join(self.output_dir, 
                                  f'sub-{subject}_searchlight_accuracy.nii.gz')
        nib.save(accuracy_map, output_path)
        print(f"  Saved searchlight map to {output_path}")
        
        return accuracy_map
    
    def roi_classification(self,
                          subject,
                          roi_mask,
                          roi_name='ROI',
                          label_column='prev_choice_n1_binary',
                          n_permutations=100):
        """
        Perform classification within a specific ROI.
        
        Parameters
        ----------
        subject : str
            Subject ID
        roi_mask : str or Nifti1Image
            ROI mask image
        roi_name : str
            Name of ROI for reporting
        label_column : str
            Column name for classification labels
        n_permutations : int
            Number of permutations for significance testing
            
        Returns
        -------
        dict
            Results dictionary
        """
        print(f"\nROI classification for subject {subject} in {roi_name}...")
        
        # Temporarily set mask to ROI
        original_mask = self.mask_img
        self.mask_img = roi_mask
        
        # Run classification
        results = self.within_subject_classification(
            subject=subject,
            label_column=label_column,
            n_permutations=n_permutations
        )
        
        # Restore original mask
        self.mask_img = original_mask
        
        if results is not None:
            results['roi_name'] = roi_name
        
        return results
    
    def group_analysis(self,
                      subjects,
                      label_column='prev_choice_n1_binary',
                      n_permutations=100):
        """
        Run classification for all subjects and perform group statistics.
        
        Parameters
        ----------
        subjects : list
            List of subject IDs
        label_column : str
            Column name for classification labels
        n_permutations : int
            Number of permutations per subject
            
        Returns
        -------
        pd.DataFrame
            Group results
        """
        print("\n" + "="*60)
        print("GROUP ANALYSIS")
        print("="*60)
        
        all_results = []
        
        for subject in subjects:
            try:
                results = self.within_subject_classification(
                    subject=subject,
                    label_column=label_column,
                    n_permutations=n_permutations
                )
                if results is not None:
                    all_results.append(results)
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
                continue
        
        if len(all_results) == 0:
            print("No subjects successfully processed")
            return None
        
        # Create summary dataframe
        summary_data = []
        for r in all_results:
            summary_data.append({
                'subject': r['subject'],
                'n_trials': r['n_trials'],
                'cv_accuracy': r['cv_accuracy_mean'],
                'cv_std': r['cv_accuracy_std'],
                'pvalue': r['permutation_pvalue'],
                'significant': r['permutation_pvalue'] < 0.05
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Group statistics
        print("\n" + "="*60)
        print("GROUP STATISTICS")
        print("="*60)
        
        accuracies = summary_df['cv_accuracy'].values
        chance_level = all_results[0]['chance_level']
        
        print(f"\nNumber of subjects: {len(summary_df)}")
        print(f"Chance level: {chance_level:.3f}")
        print(f"\nGroup mean accuracy: {accuracies.mean():.3f} ± {accuracies.std():.3f}")
        print(f"Min accuracy: {accuracies.min():.3f}")
        print(f"Max accuracy: {accuracies.max():.3f}")
        print(f"Median accuracy: {np.median(accuracies):.3f}")
        
        # One-sample t-test against chance
        t_stat, t_pval = stats.ttest_1samp(accuracies, chance_level)
        print(f"\nOne-sample t-test vs. chance ({chance_level:.3f}):")
        print(f"  t({len(accuracies)-1}) = {t_stat:.3f}, p = {t_pval:.4f}")
        
        # Effect size (Cohen's d)
        cohens_d = (accuracies.mean() - chance_level) / accuracies.std()
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        # Number of significant subjects
        n_sig = summary_df['significant'].sum()
        print(f"\nSubjects with significant decoding (p < 0.05): {n_sig}/{len(summary_df)}")
        
        # Save results
        summary_path = os.path.join(self.output_dir, 'group_classification_results.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nResults saved to {summary_path}")
        
        return summary_df


def main():
    """
    Main function to run MVPA classification.
    """
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    # Path to MVPA labels
    LABELS_PATH = "/scratch/users/gustxsr/lsa_trial_betas/mvpa_labels_binary.csv"
    
    # Output directory
    OUTPUT_DIR = "/scratch/users/gustxsr/mvpa_results"
    
    # Brain mask (optional, can use None to create from data)
    MASK_PATH = None  # or path to a mask file
    
    # Subjects to analyze
    SUBJECTS = ["001", "003", "005", "009", "011"]
    
    # Classification label column
    LABEL_COLUMN = 'prev_choice_n1_binary'  # or 'prev_choice_n1' for 4-way
    
    # Number of permutations for significance testing
    N_PERMUTATIONS = 1000
    
    # ============================================================================
    
    # Create MVPA analyzer
    mvpa = ChoiceHistoryMVPA(
        labels_path=LABELS_PATH,
        mask_img=MASK_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Run group analysis
    group_results = mvpa.group_analysis(
        subjects=SUBJECTS,
        label_column=LABEL_COLUMN,
        n_permutations=N_PERMUTATIONS
    )
    
    # Example: Single subject searchlight (commented out as it's slow)
    # accuracy_map = mvpa.searchlight_analysis(
    #     subject="001",
    #     label_column=LABEL_COLUMN,
    #     radius=5.0,
    #     n_jobs=-1
    # )
    
    # Example: ROI analysis (if you have ROI masks)
    # roi_results = mvpa.roi_classification(
    #     subject="001",
    #     roi_mask="/path/to/vmPFC_mask.nii.gz",
    #     roi_name="vmPFC",
    #     label_column=LABEL_COLUMN
    # )
    
    print("\n" + "="*60)
    print("MVPA Analysis Complete!")
    print("="*60)
    
    return group_results


if __name__ == "__main__":
    results = main()