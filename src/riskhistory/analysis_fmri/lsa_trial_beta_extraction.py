"""
LSA (Least Squares - All) Trial Beta Extraction for MVPA
========================================================

This script extracts trial-by-trial beta estimates from BIDS-formatted fMRI data
for the mixed gambles task. It's designed for subsequent MVPA analysis of choice
history encoding.

Author: Generated for NARPS mixed gambles analysis
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
import warnings
import sys
warnings.filterwarnings('ignore')


class LSATrialBetaExtractor:
    """
    Extract trial-by-trial beta estimates using Least Squares All (LSA) approach.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS directory containing preprocessed fMRI data
    derivatives_dir : str
        Path to fMRIPrep derivatives directory
    output_dir : str
        Path to save trial beta maps
    tr : float
        Repetition time in seconds
    hrf_model : str
        HRF model to use (default: 'spm')
    smoothing_fwhm : float or None
        Smoothing kernel FWHM in mm (default: None, assumes already smoothed)
    mask : str or None
        Path to brain mask (if None, will use mask from fMRIPrep)
    """
    
    def __init__(self, 
                 bids_dir,
                 derivatives_dir,
                 output_dir,
                 tr=1.0,
                 hrf_model='spm',
                 smoothing_fwhm=None,
                 mask=None,
                 high_pass=1./128,
                 standardize=False):
        
        self.bids_dir = bids_dir
        self.derivatives_dir = derivatives_dir
        self.output_dir = output_dir
        self.tr = tr
        self.hrf_model = hrf_model
        self.smoothing_fwhm = smoothing_fwhm
        self.mask = mask
        self.high_pass = high_pass
        self.standardize = standardize
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def get_subject_files(self, subject):
        """
        Get functional files and event files for a subject.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., '001')
            
        Returns
        -------
        dict
            Dictionary with runs as keys and file paths as values
        """
        sub_id = f"sub-{subject}"
        
        # Path to preprocessed functional data
        func_dir = os.path.join(self.derivatives_dir, sub_id, 'func')
        
        # Path to event files
        event_dir = os.path.join(self.bids_dir, sub_id, 'func')
        
        files = {}
        
        # Find all runs
        if not os.path.exists(func_dir):
            print(f"Warning: No functional directory found for {sub_id}")
            return files
            
        for fname in os.listdir(func_dir):
            if 'task-MGT' in fname and fname.endswith('_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'):
                # Extract run number
                run = fname.split('run-')[1].split('_')[0]
                
                # Get corresponding files
                func_file = os.path.join(func_dir, fname)
                
                # Event file
                event_file = os.path.join(event_dir, 
                                         f"{sub_id}_task-MGT_run-0{run}_events_ORIGINAL.tsv")
                
                # Confounds file
                confounds_file = os.path.join(func_dir,
                                             f"{sub_id}_task-MGT_run-{run}_desc-confounds_timeseries.tsv")
                
                # Brain mask
                if self.mask is None:
                    mask_file = os.path.join(func_dir,
                                            f"{sub_id}_task-MGT_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
                else:
                    mask_file = self.mask
                    
                if os.path.exists(event_file):
                    files[run] = {
                        'func': func_file,
                        'events': event_file,
                        'confounds': confounds_file if os.path.exists(confounds_file) else None,
                        'mask': mask_file if os.path.exists(mask_file) else None
                    }
                    
        return files
    
    def prepare_trial_events(self, events_df):
        """
        Prepare event dataframe for trial-wise GLM.
        Each trial gets a unique regressor.
        
        Parameters
        ----------
        events_df : pd.DataFrame
            Event file loaded from BIDS
            
        Returns
        -------
        pd.DataFrame
            Modified events with trial labels
        dict
            Metadata about trials (previous choice, etc.)
        """
        # Filter to only actual trials (those with responses)
        # Assuming your events file has participant_response column
        if 'participant_response' in events_df.columns:
            trial_events = events_df[events_df['participant_response'] != 'noresp'].copy()
        else:
            # If no response column, assume all are trials
            trial_events = events_df.copy()
        
        # Create unique trial labels
        trial_events['trial_label'] = [f'trial_{i:03d}' for i in range(len(trial_events))]
        
        # Prepare metadata for each trial
        trial_metadata = {}
        for idx, row in trial_events.iterrows():
            trial_label = row['trial_label']
            trial_metadata[trial_label] = {
                'onset': row['onset'],
                'duration': row.get('duration', 4.0),  # Default 4s if not specified
                'gain': row.get('gain', np.nan),
                'loss': row.get('loss', np.nan),
                'response': row.get('participant_response', 'unknown'),
                'RT': row.get('RT', np.nan)
            }
            
            # Add any other columns that might be useful
            for col in ['sv', 'entropy', 'switch', 'repeat']:
                if col in row:
                    trial_metadata[trial_label][col] = row[col]
                    
        return trial_events, trial_metadata
    
    def get_confounds(self, confounds_file, n_scans):
        """
        Load and prepare confound regressors.
        
        Parameters
        ----------
        confounds_file : str
            Path to confounds file
        n_scans : int
            Number of scans in the run
            
        Returns
        -------
        pd.DataFrame or None
            Confounds dataframe
        """
        if confounds_file is None or not os.path.exists(confounds_file):
            return None
            
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        # Select confounds to include
        confound_vars = []
        
        # Motion parameters (6 or 24)
        motion_params = [col for col in confounds_df.columns if col.startswith(('trans_', 'rot_'))]
        confound_vars.extend(motion_params)
        
        # Add derivatives if using 24-parameter model
        # motion_derivs = [col for col in confounds_df.columns if col.startswith(('trans_', 'rot_')) and '_derivative1' in col]
        # confound_vars.extend(motion_derivs)
        
        # Add motion squared and derivative squared if available
        # motion_power = [col for col in confounds_df.columns if col.startswith(('trans_', 'rot_')) and '_power2' in col]
        # confound_vars.extend(motion_power)
        
        # Physiological confounds (if available)
        for confound in ['csf', 'white_matter', 'global_signal']:
            if confound in confounds_df.columns:
                confound_vars.append(confound)
        
        # Select only available confounds
        available_confounds = [c for c in confound_vars if c in confounds_df.columns]
        
        if len(available_confounds) == 0:
            return None
            
        confounds = confounds_df[available_confounds].fillna(0)
        
        # Ensure confounds match number of scans
        if len(confounds) != n_scans:
            print(f"Warning: Confounds length ({len(confounds)}) doesn't match n_scans ({n_scans})")
            
        return confounds
    
    def create_trial_design_matrix(self, trial_events, n_scans, confounds=None):
        """
        Create design matrix with one regressor per trial.
        
        Parameters
        ----------
        trial_events : pd.DataFrame
            Trial events with trial labels
        n_scans : int
            Number of scans
        confounds : pd.DataFrame or None
            Confound regressors
            
        Returns
        -------
        pd.DataFrame
            Design matrix
        list
            Trial regressor names (for identifying beta columns)
        """
        # Create frame times
        frame_times = np.arange(n_scans) * self.tr
        
        # Prepare events for design matrix
        # Each trial needs: onset, duration, trial_type
        events_for_design = []
        
        for _, trial in trial_events.iterrows():
            events_for_design.append({
                'onset': trial['onset'],
                'duration': trial.get('RT', 0),
                'trial_type': trial['trial_label']
            })
        
        events_df = pd.DataFrame(events_for_design)
        
        # Rename confound columns to avoid conflicts with trial names or other regressors
        if confounds is not None:
            confounds_renamed = confounds.copy()
            # Add 'confound_' prefix to all confound columns to ensure uniqueness
            confounds_renamed.columns = ['confound_' + col for col in confounds.columns]
            add_reg_names = confounds_renamed.columns.tolist()
        else:
            confounds_renamed = None
            add_reg_names = None
        
        # Create design matrix
        design_matrix = make_first_level_design_matrix(
            frame_times=frame_times,
            events=events_df,
            hrf_model=self.hrf_model,
            drift_model='cosine',
            high_pass=self.high_pass,
            add_regs=confounds_renamed,
            add_reg_names=add_reg_names
        )
        print("done")
        # Check for duplicate column names
        if len(design_matrix.columns) != len(set(design_matrix.columns)):
            duplicates = [col for col in design_matrix.columns if list(design_matrix.columns).count(col) > 1]
            print(f"WARNING: Duplicate columns detected: {set(duplicates)}")
            # Remove duplicates by keeping first occurrence
            design_matrix = design_matrix.loc[:, ~design_matrix.columns.duplicated()]
        
        # Get trial regressor names (exclude drift, constant, and confounds)
        trial_regressors = [col for col in design_matrix.columns 
                           if col.startswith('trial_')]
        
        return design_matrix, trial_regressors
    
    def fit_lsa_model(self, func_file, design_matrix, mask_file=None):
        """
        Fit LSA model to extract trial betas.
        
        Parameters
        ----------
        func_file : str
            Path to functional data
        design_matrix : pd.DataFrame
            Design matrix with trial regressors
        mask_file : str or None
            Path to brain mask
            
        Returns
        -------
        dict
            Dictionary with trial labels as keys and beta maps as values
        """
        # Load functional data
        func_img = nib.load(func_file)
        
        # Create and fit FirstLevelModel
        fmri_glm = FirstLevelModel(
            t_r=self.tr,
            slice_time_ref=0.5,
            hrf_model=None,  # We already convolved in design matrix
            drift_model=None,  # Already in design matrix
            high_pass=None,  # Already in design matrix
            mask_img=mask_file,
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=self.standardize,
            noise_model='ar1',
            verbose=0,
            n_jobs=1,
            minimize_memory=False
        )
        
        # Fit the model
        fmri_glm.fit(func_img, design_matrices=design_matrix)
        
        return fmri_glm
    
    def extract_and_save_betas(self, subject, run, glm_model, trial_regressors, trial_metadata):
        """
        Extract beta maps for each trial and save them.
        
        Parameters
        ----------
        subject : str
            Subject ID
        run : str
            Run ID
        glm_model : FirstLevelModel
            Fitted GLM model
        trial_regressors : list
            List of trial regressor names
        trial_metadata : dict
            Metadata for each trial
            
        Returns
        -------
        pd.DataFrame
            Metadata dataframe with paths to beta maps
        """
        # Create output directory for this subject/run
        output_subdir = os.path.join(self.output_dir, f'sub-{subject}', f'run-{run}')
        os.makedirs(output_subdir, exist_ok=True)
        
        beta_metadata = []
        
        for trial_reg in trial_regressors:
            # Compute contrast for this trial (1 for trial, 0 for all others)
            contrast = np.zeros(len(glm_model.design_matrices_[0].columns))
            trial_idx = glm_model.design_matrices_[0].columns.get_loc(trial_reg)
            contrast[trial_idx] = 1.0
            
            # Compute beta map
            beta_map = glm_model.compute_contrast(contrast, output_type='effect_size')
            
            # Save beta map
            beta_filename = f'sub-{subject}_run-{run}_{trial_reg}_beta.nii.gz'
            beta_path = os.path.join(output_subdir, beta_filename)
            nib.save(beta_map, beta_path)
            
            # Store metadata
            meta = trial_metadata[trial_reg].copy()
            meta['subject'] = subject
            meta['run'] = run
            meta['trial_label'] = trial_reg
            meta['beta_path'] = beta_path
            beta_metadata.append(meta)
            
        return pd.DataFrame(beta_metadata)
    
    def process_subject(self, subject, save_metadata=True):
        """
        Process all runs for a subject and extract trial betas.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., '001')
        save_metadata : bool
            Whether to save metadata CSV
            
        Returns
        -------
        pd.DataFrame
            Metadata for all trials
        """
        print(f"\n{'='*60}")
        print(f"Processing subject {subject}")
        print(f"{'='*60}")
        
        # Get files for this subject
        files = self.get_subject_files(subject)
        
        if len(files) == 0:
            print(f"No files found for subject {subject}")
            return None
        
        all_metadata = []
        
        for run, file_dict in sorted(files.items()):
            print(f"\nProcessing run {run}...")
            
            # Load events
            events_df = pd.read_csv(file_dict['events'], sep='\t')
            
            # Prepare trial events
            trial_events, trial_metadata = self.prepare_trial_events(events_df)
            print(f"  Found {len(trial_events)} trials")
            
            # Get number of scans
            func_img = nib.load(file_dict['func'])
            n_scans = func_img.shape[-1]
            print(f"  Number of scans: {n_scans}")
            
            # Load confounds
            confounds = self.get_confounds(file_dict['confounds'], n_scans)
            if confounds is not None:
                print(f"  Using {len(confounds.columns)} confound regressors")
            
            # Create design matrix
            print("  Creating design matrix...")
            design_matrix, trial_regressors = self.create_trial_design_matrix(
                trial_events, n_scans, confounds
            )
            print(f"  Design matrix shape: {design_matrix.shape}")
            print(f"  Number of trial regressors: {len(trial_regressors)}")
            print(f"  Number of confound regressors: {len([c for c in design_matrix.columns if c.startswith('confound_')])}")
            print(f"  Number of drift regressors: {len([c for c in design_matrix.columns if 'drift' in c.lower() or 'cosine' in c.lower()])}")
            
            # Verify no duplicate columns
            if len(design_matrix.columns) != len(set(design_matrix.columns)):
                duplicates = [col for col in design_matrix.columns if list(design_matrix.columns).count(col) > 1]
                raise ValueError(f"Design matrix has duplicate columns: {set(duplicates)}")
            
            # Fit LSA model
            print("  Fitting LSA model...")
            glm_model = self.fit_lsa_model(
                file_dict['func'],
                design_matrix,
                file_dict['mask']
            )
            
            # Extract and save betas
            print("  Extracting and saving beta maps...")
            run_metadata = self.extract_and_save_betas(
                subject, run, glm_model, trial_regressors, trial_metadata
            )
            
            all_metadata.append(run_metadata)
            print(f"  Completed run {run}: {len(run_metadata)} trial betas saved")
        
        # Combine metadata from all runs
        subject_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # Save metadata
        if save_metadata:
            metadata_path = os.path.join(self.output_dir, f'sub-{subject}', 
                                        f'sub-{subject}_trial_metadata.csv')
            subject_metadata.to_csv(metadata_path, index=False)
            print(f"\nMetadata saved to {metadata_path}")
        
        print(f"\nCompleted subject {subject}: {len(subject_metadata)} total trial betas")
        
        return subject_metadata
    
    def process_all_subjects(self, subjects):
        """
        Process multiple subjects.
        
        Parameters
        ----------
        subjects : list
            List of subject IDs
            
        Returns
        -------
        pd.DataFrame
            Combined metadata for all subjects
        """
        all_metadata = []
        
        for subject in subjects:
            try:
                subject_metadata = self.process_subject(subject)
                if subject_metadata is not None:
                    all_metadata.append(subject_metadata)
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
                continue
        
        # Combine all metadata
        if len(all_metadata) > 0:
            combined_metadata = pd.concat(all_metadata, ignore_index=True)
            
            # Save combined metadata
            combined_path = os.path.join(self.output_dir, 'all_subjects_trial_metadata.csv')
            combined_metadata.to_csv(combined_path, index=False)
            print(f"\n{'='*60}")
            print(f"All subjects processed!")
            print(f"Combined metadata saved to {combined_path}")
            print(f"Total trials: {len(combined_metadata)}")
            print(f"{'='*60}")
            
            return combined_metadata
        else:
            print("No subjects were successfully processed")
            return None


def main():
    """
    Main function to run LSA trial beta extraction.
    """
    # ============================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ============================================================================
    
    # Path to BIDS directory (where event files are)
    BIDS_DIR = "/oak/stanford/groups/russpold/data/openneuro_fitlins/input/ds001734"
    
    # Path to fMRIPrep derivatives (preprocessed functional data)
    DERIVATIVES_DIR = "/oak/stanford/groups/russpold/data/openneuro_fitlins/fmriprep/ds001734"
    
    # Output directory for trial betas
    OUTPUT_DIR = "/scratch/users/gustxsr/lsa_trial_betas"
    
    # Subjects to process (update this list)
    if sys.argv[1] == "equalIndifference":
        SUBJECTS = [
            # "001",
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
    
    # Scanning parameters
    TR = 1.0  # Repetition time in seconds
    
    # GLM parameters
    HRF_MODEL = 'spm'  # Can be 'spm', 'glover', 'spm + derivative', etc.
    SMOOTHING_FWHM = None  # Set to None if already smoothed, otherwise e.g., 5.0
    HIGH_PASS = 1.0/128  # High-pass filter cutoff in Hz
    
    # ============================================================================
    
    # Create extractor
    extractor = LSATrialBetaExtractor(
        bids_dir=BIDS_DIR,
        derivatives_dir=DERIVATIVES_DIR,
        output_dir=OUTPUT_DIR,
        tr=TR,
        hrf_model=HRF_MODEL,
        smoothing_fwhm=SMOOTHING_FWHM,
        high_pass=HIGH_PASS,
        standardize=False
    )
    
    # Process all subjects
    metadata = extractor.process_all_subjects(SUBJECTS)
    
    print("\nLSA trial beta extraction complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return metadata


if __name__ == "__main__":
    metadata = main()
