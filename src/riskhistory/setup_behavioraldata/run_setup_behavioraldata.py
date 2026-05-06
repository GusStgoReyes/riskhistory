# =============================================================================
# Choice History Under Risk: Run the behavioral data setup script
# =============================================================================

from riskhistory.config import (narps_behavioral_data_pth,
                                sheng2020_behavioral_data_pth,
                                sheng2020_pupil_data_pth,
                                behavioral_data_pth,
                                subject_statistics_pth,
                                trial_exclusion_counts_pth,
                                trial_exclusion_summary_pth,
                                behavioral_summary_pth)

from riskhistory.setup_behavioraldata.utils_behavioraldata import (
    prepare_narps_behavioral_data,
    prepare_sheng2020_behavioral_data,
    add_exclude_trials,
    compute_exclusion_counts,
    summarize_exclusion_counts,
    compute_subject_statistics,
    exclude_participants_by_criteria
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
MIN_PREV_VALID_TRIALS = 2  # Minimum previous valid trials for choice history

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


def generate_summary(all_behav_df, subject_stats_df,
                     min_rt, max_rt, min_prev_valid_trials,
                     max_excluded_trials_pct, min_acceptance_rate, max_acceptance_rate,
                     min_mean_rt, max_mean_rt, manually_excluded):
    lines = []
    lines.append("=" * 70)
    lines.append("BEHAVIORAL DATA SETUP SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    lines.append("\nEXCLUSION CRITERIA APPLIED:")
    lines.append("-" * 40)
    lines.append("Trial-level:")
    lines.append(f"  - Response time: {min_rt}s < RT < {max_rt}s")
    lines.append(f"  - Minimum previous valid trials: {min_prev_valid_trials}")
    lines.append(f"  - Valid gain/loss pair required")
    lines.append("Participant-level:")
    lines.append(f"  - Max excluded trials: {max_excluded_trials_pct*100:.0f}%")
    lines.append(f"  - Acceptance rate: {min_acceptance_rate*100:.0f}% - {max_acceptance_rate*100:.0f}%")
    lines.append(f"  - Mean RT: {min_mean_rt}s - {max_mean_rt}s")
    if manually_excluded:
        lines.append(f"  - Manually excluded: {manually_excluded}")

    total_subjects = len(subject_stats_df)
    total_excluded = subject_stats_df['excluded'].sum()
    total_included = total_subjects - total_excluded

    lines.append("\n" + "=" * 70)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total subjects: {total_subjects}")
    lines.append(f"Included subjects: {total_included}")
    lines.append(f"Excluded subjects: {total_excluded}")

    for dataset in subject_stats_df['dataset'].unique():
        ds_stats = subject_stats_df[subject_stats_df['dataset'] == dataset]
        ds_included = ds_stats[~ds_stats['excluded']].copy()
        ds_included["num_valid_trials"] = ds_included["n_trials"] - ds_included["n_excluded_trials"]

        n_total = len(ds_stats)
        n_excluded = ds_stats['excluded'].sum()
        n_included = n_total - n_excluded

        ds_behav = all_behav_df[all_behav_df['dataset'] == dataset]
        n_trials = len(ds_behav)
        n_valid_trials = len(ds_behav[~ds_behav['exclude']])

        lines.append(f"\n{dataset.upper()}")
        lines.append("-" * 40)
        lines.append(f"Subjects: {n_included} included / {n_total} total ({n_excluded} excluded)")
        lines.append(f"Trials: {n_valid_trials} valid / {n_trials} total ({100*n_valid_trials/n_trials:.1f}%)")

        if len(ds_included) > 0:
            lines.append(f"Acceptance rate: M={ds_included['acceptance_rate'].mean():.3f}, "
                         f"SD={ds_included['acceptance_rate'].std():.3f}, "
                         f"range=[{ds_included['acceptance_rate'].min():.3f}, {ds_included['acceptance_rate'].max():.3f}]")
            lines.append(f"Response time: M={ds_included['mean_rt'].mean():.3f}s, "
                         f"SD={ds_included['mean_rt'].std():.3f}s, "
                         f"range=[{ds_included['mean_rt'].min():.3f}, {ds_included['mean_rt'].max():.3f}]s")
            lines.append(f"Trials: M={ds_included['num_valid_trials'].mean():.3f}, "
                         f"SD={ds_included['num_valid_trials'].std():.3f}, "
                         f"range=[{ds_included['num_valid_trials'].min():.3f}, {ds_included['num_valid_trials'].max():.3f}]")

    excluded_subs = subject_stats_df[subject_stats_df['excluded']]
    if len(excluded_subs) > 0:
        lines.append("\n" + "=" * 70)
        lines.append("EXCLUDED PARTICIPANTS")
        lines.append("=" * 70)
        for _, row in excluded_subs.iterrows():
            lines.append(f"  {row['dataset']} sub-{row['sub']:02d}: {row['exclusion_reason']}")

    lines.append("\n" + "=" * 70)
    lines.append("OUTPUT FILES")
    lines.append("=" * 70)
    lines.append(f"Behavioral data: {behavioral_data_pth}")
    lines.append(f"Subject statistics: {subject_statistics_pth}")
    lines.append(f"This summary: {behavioral_summary_pth}")
    lines.append("=" * 70)

    return "\n".join(lines)


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

    # =========================================================================
    # Step 9: Generate and save summary
    # =========================================================================
    summary_text = generate_summary(
        all_behav_df, subject_stats_df,
        min_rt=MIN_RT, max_rt=MAX_RT, min_prev_valid_trials=MIN_PREV_VALID_TRIALS,
        max_excluded_trials_pct=MAX_EXCLUDED_TRIALS_PCT,
        min_acceptance_rate=MIN_ACCEPTANCE_RATE, max_acceptance_rate=MAX_ACCEPTANCE_RATE,
        min_mean_rt=MIN_MEAN_RT, max_mean_rt=MAX_MEAN_RT,
        manually_excluded=MANUALLY_EXCLUDED_PARTICIPANTS
    )

    print(summary_text)

    Path(behavioral_summary_pth).parent.mkdir(parents=True, exist_ok=True)
    with open(behavioral_summary_pth, 'w') as f:
        f.write(summary_text)

    print(f"\nSummary saved to {behavioral_summary_pth}")
