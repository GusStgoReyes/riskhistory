# =============================================================================
# Run the behavioral data setup script
# =============================================================================

from riskhistory.config import (narps_behavioral_data_pth,
                                sheng2020_behavioral_data_pth,
                                sheng2020_pupil_data_pth,
                                behavioral_data_pth,
                                subject_statistics_pth,
                                trial_exclusion_counts_pth,
                                trial_exclusion_summary_pth,
                                participant_exclusion_summary_pth,
                                final_sample_summary_pth)

from riskhistory.setup_behavioraldata.utils_behavioraldata import (
    prepare_narps_behavioral_data,
    prepare_sheng2020_behavioral_data,
    add_exclude_trials,
    compute_exclusion_counts,
    summarize_exclusion_counts,
    compute_subject_statistics,
    exclude_participants_by_criteria,
    summarize_participant_exclusions,
    summarize_final_sample
)

import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# Exclusion criteria
# =============================================================================
# Trial-level exclusion criteria
MIN_RT = 0.2  # Minimum response time threshold (seconds)
MAX_RT = 3.8  # Maximum response time threshold (seconds)
MIN_PREV_VALID_TRIALS = 1  # Minimum previous valid trials for choice history

# Participant-level exclusion criteria
MAX_EXCLUDED_TRIALS_PCT = 0.30  # Exclude participants with >30% excluded trials
MIN_ACCEPTANCE_RATE = 0.10  # Exclude participants with <10% acceptance rate
MAX_ACCEPTANCE_RATE = 0.90  # Exclude participants with >90% acceptance rate
MIN_MEAN_RT = 0.3  # Exclude participants with mean RT < 0.3s
MAX_MEAN_RT = 3.5  # Exclude participants with mean RT > 3.5s

# Additional fixes/exclusions:
#   1. Exclude Subject 29 in Sheng2020 (time series for Run 2 is missing a trial indicator)
#   2. Fix Subject 56's responses in NARPS (they were all shifted by one trial)

# Manually excluded participants (e.g., due to data quality issues)
MANUALLY_EXCLUDED_PARTICIPANTS = [("sheng2020", 29)]  # (dataset, sub)


if __name__ == "__main__":
    # =========================================================================
    # Step 1: Load raw behavioral data
    # =========================================================================
    print("Loading raw behavioral data...")
    narps_df = pd.read_csv(narps_behavioral_data_pth)
    sheng2020_response_df = pd.read_csv(sheng2020_behavioral_data_pth)
    sheng2020_pupil_df = pd.read_csv(sheng2020_pupil_data_pth)

    # =========================================================================
    # Step 2: Prepare behavioral data for each dataset
    # =========================================================================
    print("Preparing NARPS behavioral data...")
    narps_behav_df = prepare_narps_behavioral_data(
        narps_df,
        min_rt=MIN_RT,
        max_rt=MAX_RT
    )

    print("Preparing Sheng2020 behavioral data...")
    sheng2020_behav_df = prepare_sheng2020_behavioral_data(
        sheng2020_response_df,
        sheng2020_pupil_df,
        min_rt=MIN_RT,
        max_rt=MAX_RT
    )

    # =========================================================================
    # Step 3: Merge datasets
    # =========================================================================
    print("Merging datasets...")
    all_behav_df = pd.concat([narps_behav_df, sheng2020_behav_df], ignore_index=True)

    # =========================================================================
    # Step 4: Add trial exclusion column
    # =========================================================================
    print("Adding trial exclusion flags...")
    all_behav_df = add_exclude_trials(
        all_behav_df,
        min_rt=MIN_RT,
        max_rt=MAX_RT,
        min_prev_valid_trials=MIN_PREV_VALID_TRIALS
    )

    # =========================================================================
    # Step 5: Compute trial exclusion counts per subject (before participant exclusion)
    # =========================================================================
    print("Computing trial exclusion counts...")
    trial_exclusion_df = compute_exclusion_counts(
        all_behav_df,
        min_rt=MIN_RT,
        max_rt=MAX_RT,
        min_prev_valid_trials=MIN_PREV_VALID_TRIALS
    )
    trial_exclusion_summary_df = summarize_exclusion_counts(trial_exclusion_df)

    # =========================================================================
    # Step 6: Compute subject-level statistics (before participant exclusion)
    # =========================================================================
    print("Computing subject-level statistics...")
    subject_stats_df = compute_subject_statistics(all_behav_df)

    # =========================================================================
    # Step 7: Exclude participants based on criteria
    # =========================================================================
    print("Applying participant exclusion criteria...")
    all_behav_df, subject_stats_df = exclude_participants_by_criteria(
        all_behav_df,
        subject_stats_df,
        max_excluded_trials_pct=MAX_EXCLUDED_TRIALS_PCT,
        min_acceptance_rate=MIN_ACCEPTANCE_RATE,
        max_acceptance_rate=MAX_ACCEPTANCE_RATE,
        min_mean_rt=MIN_MEAN_RT,
        max_mean_rt=MAX_MEAN_RT,
        manually_excluded=MANUALLY_EXCLUDED_PARTICIPANTS
    )

    # =========================================================================
    # Step 7b: Summarize participant exclusions and the final retained sample
    # =========================================================================
    print("Summarizing participant exclusions...")
    participant_exclusion_summary_df = summarize_participant_exclusions(subject_stats_df)

    print("Summarizing final sample...")
    final_sample_summary_df = summarize_final_sample(subject_stats_df)

    # =========================================================================
    # Step 8: Save outputs
    # =========================================================================
    print(f"Saving behavioral data to {behavioral_data_pth}...")
    all_behav_df.to_csv(behavioral_data_pth, index=False)

    print(f"Saving subject statistics to {subject_statistics_pth}...")
    subject_stats_df.to_csv(subject_statistics_pth, index=False)

    print(f"Saving trial exclusion counts to {trial_exclusion_counts_pth}...")
    trial_exclusion_counts_pth.parent.mkdir(parents=True, exist_ok=True)
    trial_exclusion_df.to_csv(trial_exclusion_counts_pth, index=False)

    print(f"Saving trial exclusion summary to {trial_exclusion_summary_pth}...")
    trial_exclusion_summary_df.to_csv(trial_exclusion_summary_pth, index=False)

    print(f"Saving participant exclusion summary to {participant_exclusion_summary_pth}...")
    participant_exclusion_summary_df.to_csv(participant_exclusion_summary_pth, index=False)

    print(f"Saving final sample summary to {final_sample_summary_pth}...")
    final_sample_summary_df.to_csv(final_sample_summary_pth, index=False)
