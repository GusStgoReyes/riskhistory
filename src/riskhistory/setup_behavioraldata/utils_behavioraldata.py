# =============================================================================
# Behavioral Data Preparation Functions
# =============================================================================
# Columns that should be present in the final dataframe:
# 'sub': subject identifier
# 'trial': trial number
# 'trial_run': trial number within the run
# 'accept': choice made by the subject (0 or 1)
# 'response_raw': original coding of the response (e.g. for NARPS: 1, 2, 3, 4, or nan)
# 'gain': gain amount for the trial
# 'loss': loss amount for the trial
# 'prev_gain' : gain amount from the previous trial
# 'prev_loss' : loss amount from the previous trial
# 'gain_centered' : gain amount centered within each dataset
# 'loss_centered' : loss amount centered within each dataset
# 'prev_gain_centered' : previous gain amount centered within each dataset
# 'prev_loss_centered' : previous loss amount centered within each dataset
# 'rt': reaction time for the trial
# 'dataset': dataset identifier (equalIndifference, equalRange, sheng2020)
# 'choice': choice made on the current trial (1 or -1)
# 'choice_prev[N]': choice made N trials ago (1 or -1) for N in 1 to 6 (within a run)
# 'switch_button' : whether the subject switched their button press from the previous trial (1 for switch, 0 for stay, NAN)
# 'repeat' : whether the subject repeated their response from the previous trial (1 for repeat, 0 for switch, NAN)
# 'switch' : whether the subject switched their choice from the previous trial (1 for switch, 0 for stay, NAN)
# 'streak' : how many trials back to find a different choice (0 for first trial, 1 if previous trial is different, 2 if previous two trials are the same and the third trial back is different, etc.)
# 'logstreak_on_switch' : log(streak) when switch==1, 0 otherwise
# 'logstreak_on_repeat' : log(streak) when repeat==1, 0 otherwise
# 'exclude' : whether the trial should be excluded based on some criteria (True for exclude, False for include)

import pandas as pd
import numpy as np

STIMULI_CENTER = {
    "equalIndifference" : {"gain": 25, "loss": 12.5},
    "equalRange" : {"gain": 12.5, "loss": 12.5},
    "sheng2020" : {"gain": 5.5, "loss": 5.5}
}


def _final_columns(N_max=6):
    return (
        ['sub', 'trial', 'trial_run', 'run', 'accept', 'response_raw', 'gain', 'loss',
         'gain_centered', 'loss_centered', 'prev_gain_centered', 'prev_loss_centered',
         'rt', 'dataset', 'choice'] +
        [f'choice_prev{N}' for N in range(1, N_max + 1)] +
        ['switch_button', 'repeat', 'switch', 'prev_gain', 'prev_loss',
         'streak', 'logstreak_on_switch', 'logstreak_on_repeat']
    )


def add_prev_columns(df):
    for col in ['gain', 'loss', 'gain_centered', 'loss_centered']:
        df[f'prev_{col}'] = df.groupby(['sub', 'run'])[col].shift(1)
    return df


def fix_sub_56(df):
    """
    For subject 56 of NARPS_equalRange dataset, flip the responses and remap them.
    """
    int_to_label = {
        1.0: "strongly_reject",
        2.0: "weakly_reject",
        3.0: "weakly_accept",
        4.0: "strongly_accept"
    }

    mask = df["sub"] == 56
    if mask.any():
        df.loc[mask, "response_int"] = 5 - df.loc[mask, "response_int"]
        df.loc[mask, "accept"] = df.loc[mask, "response_int"].isin([3.0, 4.0]).astype(int)
        df.loc[mask, "participant_response"] = df.loc[mask, "response_int"].map(int_to_label)

    return df


def new_row_trial(group, trial, stim):
    """
    (Necessary for Sheng2020) Create a new row for the trial mapping.
    """
    return {
        "Subject": group["Subject"].iloc[0],
        "Run": group["Run"].iloc[0],
        "trial": trial,
        "StimulusCode": stim
    }


def get_trial_ordering(group):
    """
    (Necessary for Sheng2020) Given trial ordering for a Subject/Run,
    create a mapping of trial numbers to stimulus codes.
    """
    trial, stim = None, None
    none_stim_count = 1
    rows = []
    for code in group["StimulusName"]:
        if code <= 100:
            if trial is not None:
                rows.append(new_row_trial(group, trial, stim=None))
            trial = code
        elif code > 100:
            if stim is not None:
                trial = f"noTrial_{none_stim_count}"
                rows.append(new_row_trial(group, trial, stim))
                none_stim_count += 1
            stim = code

        if trial is not None and stim is not None:
            rows.append(new_row_trial(group, trial, stim))
            trial, stim = None, None
    if stim is not None:
        trial = f"noTrial_{none_stim_count}"
        rows.append(new_row_trial(group, trial, stim))

    return pd.DataFrame(rows)


def add_gain_loss(df):
    """
    (Necessary for Sheng2020) Given a dataframe with StimulusCode, add Gain and Loss columns.
    """
    df["Loss"] = df["StimulusCode"] % 100
    df["Gain"] = df["StimulusCode"] // 100
    return df


def create_choice_columns(df, button_column='response_int', N_max=6):
    """
    Create choice and choice history columns in the dataframe.
    """
    df['choice'] = df['accept'].replace({0: -1, 1: 1})

    for N in range(1, N_max + 1):
        df[f'choice_prev{N}'] = df.groupby(['sub', 'run'])['choice'].shift(N)

    df[[f'choice_prev{N}' for N in range(1, N_max + 1)]] = (
        df[[f'choice_prev{N}' for N in range(1, N_max + 1)]].fillna(0)
    )

    df['switch_button'] = (df.groupby(['sub', 'run'])[button_column]
                            .transform(lambda x: np.where(
                                x.isna() | x.shift(1).isna(),
                                np.nan,
                                (x != x.shift(1)).astype(int)
                            )))

    df['repeat'] = (df.groupby(['sub', 'run'])["accept"]
                            .transform(lambda x: np.where(
                                x.isna() | x.shift(1).isna(),
                                np.nan,
                                (x == x.shift(1)).astype(int)
                            )))

    df['switch'] = df['repeat'].apply(lambda x: np.nan if pd.isna(x) else (0 if x == 1 else 1))

    return df


def calculate_streak(group):
    """Calculate streak: how many consecutive identical choices precede the current trial."""
    streak = [0]
    accept = group['accept'].values
    prev_streak = 0
    for i in range(1, len(accept)):
        if pd.isna(accept[i]) or pd.isna(accept[i-1]):
            streak.append(np.nan)
            prev_streak = 0
        elif i == 1:
            streak.append(1)
            prev_streak = 1
        else:
            if accept[i-1] == accept[i-2]:
                prev_streak += 1
                streak.append(prev_streak)
            else:
                prev_streak = 1
                streak.append(prev_streak)

    return pd.Series(streak, index=group.index)


def add_logstreak_columns(df):
    """
    Add log streak columns to the dataframe.
    """
    streaks = df.groupby(['sub', 'run'], group_keys=False).apply(calculate_streak)
    df["streak"] = streaks

    df['logstreak_on_switch'] = np.where(df['switch'] == 1,
                                         np.log(df['streak'] + 1e-10),
                                         0)
    df['logstreak_on_repeat'] = np.where(df['repeat'] == 1,
                                         np.log(df['streak'] + 1e-10),
                                         0)
    return df


def add_centered_columns(df, dataset_column='dataset'):
    """
    Add gain_centered and loss_centered columns based on dataset.
    """
    df['gain_centered'] = df['gain'].copy()
    df['loss_centered'] = df['loss'].copy()

    for dataset_name, centers in STIMULI_CENTER.items():
        mask = df[dataset_column] == dataset_name
        df.loc[mask, 'gain_centered'] = df.loc[mask, 'gain'] - centers['gain']
        df.loc[mask, 'loss_centered'] = df.loc[mask, 'loss'] - centers['loss']

    return df


def prepare_narps_behavioral_data(df, min_rt=0.2, max_rt=3.8, N_max=6):
    """
    Prepare NARPS behavioral data for analysis.
    """
    df = fix_sub_56(df)

    df['trial'] = np.arange(0, len(df)) % 256 + 1
    df['trial_run'] = np.arange(0, len(df)) % 64 + 1

    df = df.rename(columns={'RT': 'rt', 'condition': 'dataset'})

    # Vectorized RT validity and response coding
    invalid_rt = (df['rt'] < min_rt) | (df['rt'] > max_rt)
    original_response = df['response_int'].copy()
    df['response_raw'] = original_response

    df['accept'] = np.where(invalid_rt, np.nan,
                   np.where(original_response.isin([3.0, 4.0]), 1.0,
                   np.where(original_response.isin([1.0, 2.0]), 0.0, np.nan)))

    df['response_int'] = np.where(invalid_rt, np.nan,
                         np.where(original_response.isin([1.0, 2.0, 3.0, 4.0]), original_response, np.nan))

    df = create_choice_columns(df, button_column='response_int', N_max=N_max)
    df = add_centered_columns(df, dataset_column='dataset')
    df = add_prev_columns(df)
    df = add_logstreak_columns(df)

    return df[_final_columns(N_max)]


def prepare_sheng2020_behavioral_data(df_response, df_pupil, min_rt=0.2, max_rt=3.8, N_max=6):
    """
    Prepare Sheng2020 behavioral data for analysis.
        - df_response: DataFrame containing the response data (trials are not ordered)
        - df_pupil: DataFrame containing the pupil data (used to get trial order)
    """
    df_pupil = df_pupil[["Subject", "Run", "StimulusName"]]
    df_pupil = df_pupil.drop_duplicates(subset=["Subject", "Run", "StimulusName"]).reset_index(drop=True)

    trial_df = (
        df_pupil.groupby(["Subject", "Run"])[["Subject", "Run", "StimulusName"]]
        .apply(get_trial_ordering)
        .reset_index(drop=True)
    )

    trial_df = add_gain_loss(trial_df)
    trial_df = trial_df.drop("StimulusCode", axis=1)

    df = trial_df.merge(df_response, on=["Subject", "Run", "Gain", "Loss"], how="left")

    df["RT"] = df["RT"] / 1000

    df["trial_run"] = df["trial"] + 1
    df["trial"] = df["trial"] + 1 + 100 * (df["Run"] - 1)

    df = df.rename(columns={"Subject": "sub", "Run": "run", "Gain": "gain",
                             "Loss": "loss", "RT": "rt", "Response": "accept"})

    df['dataset'] = 'sheng2020'

    # Vectorized RT validity and response coding
    invalid_rt = (df['rt'] < min_rt) | (df['rt'] > max_rt)
    df['response_raw'] = df['accept'].copy()
    df['accept'] = np.where(invalid_rt, np.nan,
                   np.where(df['accept'].isin([0, 1, 2, 3, 4]), df['accept'].astype(float), np.nan))

    df = create_choice_columns(df, button_column='accept', N_max=N_max)
    df = add_centered_columns(df, dataset_column='dataset')
    df = add_prev_columns(df)
    df = add_logstreak_columns(df)

    return df[_final_columns(N_max)]


def add_exclude_trials(df, min_rt=0.2, max_rt=3.8, min_prev_valid_trials=2):
    df['exclude'] = False

    df.loc[(df['rt'] < min_rt) | (df['rt'] > max_rt), 'exclude'] = True

    choice_histories = [df[f'choice_prev{N}'] == 0 for N in range(1, min_prev_valid_trials + 1)]
    df.loc[np.logical_or.reduce(choice_histories), 'exclude'] = True

    df.loc[df['gain'].isna() | df['loss'].isna(), 'exclude'] = True

    return df


def compute_exclusion_counts(df, min_rt=0.2, max_rt=3.8, min_prev_valid_trials=2):
    """
    Count trials excluded by each criterion independently, per subject per dataset.

    Criteria are evaluated independently (not cumulatively), so a trial
    excluded by multiple criteria is counted once under each. The one
    exception is n_no_response vs n_excluded_min_rt: no-response trials
    (rt == 0) are counted only under n_no_response, so the two columns do
    not overlap. Note that add_exclude_trials still removes those trials
    via its rt < min_rt rule.

    Returns a DataFrame with columns:
        dataset, sub, n_trials, n_no_response, n_excluded_min_rt,
        n_excluded_max_rt, n_excluded_min_prev_valid,
        n_excluded_missing_gainloss, n_excluded_total
    """
    rows = []
    for (dataset, sub), group in df.groupby(["dataset", "sub"]):
        n_trials = len(group)

        n_no_response = (group["response_raw"].isna()).sum()
        # No-response trials are coded rt == 0 (or NaN), and would otherwise be
        # counted a second time under the min_rt criterion, so require rt > 0.
        n_min_rt = ((group["rt"] > 0) & (group["rt"] < min_rt)).sum()
        n_max_rt = (group["rt"] > max_rt).sum()

        prev_flags = [group[f"choice_prev{N}"] == 0 for N in range(1, min_prev_valid_trials + 1)]
        n_min_prev = np.logical_or.reduce(prev_flags).sum()

        n_missing_gainloss = (group["gain"].isna() | group["loss"].isna()).sum()

        n_total = group["exclude"].sum()

        rows.append({
            "dataset": dataset,
            "sub": sub,
            "n_trials": n_trials,
            "n_no_response": int(n_no_response),
            "n_excluded_min_rt": int(n_min_rt),
            "n_excluded_max_rt": int(n_max_rt),
            "n_excluded_min_prev_valid": int(n_min_prev),
            "n_excluded_missing_gainloss": int(n_missing_gainloss),
            "n_excluded_total": int(n_total),
        })

    return pd.DataFrame(rows)


def summarize_exclusion_counts(exclusion_counts_df):
    """
    Summarize per-subject exclusion counts to dataset level.

    Returns a DataFrame with columns:
        dataset, n_subjects,
        min/max of n_no_response, n_excluded_min_rt, n_excluded_max_rt,
        n_excluded_min_prev_valid, n_excluded_missing_gainloss, n_excluded_total
    """
    criteria = ["n_no_response", "n_excluded_min_rt", "n_excluded_max_rt", "n_excluded_min_prev_valid",
                "n_excluded_missing_gainloss", "n_excluded_total"]

    rows = []
    for dataset, group in exclusion_counts_df.groupby("dataset"):
        row = {"dataset": dataset, "n_subjects": len(group)}
        for col in criteria:
            row[f"mean_{col}"] = group[col].mean()
            row[f"sd_{col}"] = group[col].std()
        rows.append(row)

    return pd.DataFrame(rows)


def exclude_participants(df, participants_to_exclude):
    """
    Exclude entire participants from the dataframe.
    """
    return df[~df['sub'].isin(participants_to_exclude)].reset_index(drop=True)


def compute_subject_statistics(df):
    """
    Compute per-subject statistics for inclusion in paper and for exclusion decisions.

    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data with 'sub', 'dataset', 'accept', 'rt', 'exclude' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per subject containing:
        - sub: subject identifier
        - dataset: dataset identifier
        - n_trials: total number of trials
        - n_valid_trials: number of non-excluded trials
        - n_excluded_trials: number of excluded trials
        - excluded_trials_pct: percentage of excluded trials
        - acceptance_rate: proportion of accept responses (on valid trials)
        - mean_rt: mean response time (on valid trials)
        - std_rt: standard deviation of response time (on valid trials)
        - median_rt: median response time (on valid trials)
        - n_missing_responses: number of trials with missing accept values
    """
    stats = []

    for (sub, dataset), group in df.groupby(['sub', 'dataset']):
        n_trials = len(group)
        n_excluded = group['exclude'].sum()
        n_valid = n_trials - n_excluded

        valid_trials = group[~group['exclude']]

        n_missing = valid_trials['accept'].isna().sum()

        valid_responses = valid_trials['accept'].dropna()
        acceptance_rate = valid_responses.mean() if len(valid_responses) > 0 else np.nan

        valid_rt = valid_trials.loc[valid_trials['accept'].notna(), 'rt']
        mean_rt = valid_rt.mean() if len(valid_rt) > 0 else np.nan
        std_rt = valid_rt.std() if len(valid_rt) > 0 else np.nan
        median_rt = valid_rt.median() if len(valid_rt) > 0 else np.nan

        stats.append({
            'sub': sub,
            'dataset': dataset,
            'n_trials': n_trials,
            'n_valid_trials': n_valid,
            'n_excluded_trials': n_excluded,
            'excluded_trials_pct': n_excluded / n_trials if n_trials > 0 else np.nan,
            'acceptance_rate': acceptance_rate,
            'mean_rt': mean_rt,
            'std_rt': std_rt,
            'median_rt': median_rt,
            'n_missing_responses': n_missing,
        })

    return pd.DataFrame(stats)


def exclude_participants_by_criteria(
    df,
    subject_stats_df,
    max_excluded_trials_pct=0.30,
    min_acceptance_rate=0.10,
    max_acceptance_rate=0.90,
    min_mean_rt=0.3,
    max_mean_rt=3.5,
    manually_excluded=None
):
    """
    Exclude participants based on multiple criteria and add exclusion flags to stats.

    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data.
    subject_stats_df : pd.DataFrame
        Subject statistics from compute_subject_statistics().
    max_excluded_trials_pct : float
        Exclude if excluded_trials_pct > this value.
    min_acceptance_rate : float
        Exclude if acceptance_rate < this value.
    max_acceptance_rate : float
        Exclude if acceptance_rate > this value.
    min_mean_rt : float
        Exclude if mean_rt < this value.
    max_mean_rt : float
        Exclude if mean_rt > this value.
    manually_excluded : list of tuples, optional
        List of (dataset, sub) tuples to manually exclude.

    Returns
    -------
    df : pd.DataFrame
        Behavioral data with excluded participants removed.
    subject_stats_df : pd.DataFrame
        Updated stats with 'excluded', 'exclusion_reason', and one boolean
        column per criterion ('excluded_high_trial_exclusion_pct',
        'excluded_low_acceptance_rate', 'excluded_high_acceptance_rate',
        'excluded_low_mean_rt', 'excluded_high_mean_rt', 'excluded_manual').
    """
    if manually_excluded is None:
        manually_excluded = []

    criterion_cols = [
        'excluded_high_trial_exclusion_pct',
        'excluded_low_acceptance_rate',
        'excluded_high_acceptance_rate',
        'excluded_low_mean_rt',
        'excluded_high_mean_rt',
        'excluded_manual',
    ]

    subject_stats_df = subject_stats_df.copy()
    subject_stats_df['excluded'] = False
    subject_stats_df['exclusion_reason'] = ''
    for col in criterion_cols:
        subject_stats_df[col] = False

    for idx, row in subject_stats_df.iterrows():
        reasons = []

        if row['excluded_trials_pct'] > max_excluded_trials_pct:
            reasons.append(f"excluded_trials_pct={row['excluded_trials_pct']:.2f}")
            subject_stats_df.loc[idx, 'excluded_high_trial_exclusion_pct'] = True

        if pd.notna(row['acceptance_rate']):
            if row['acceptance_rate'] < min_acceptance_rate:
                reasons.append(f"acceptance_rate={row['acceptance_rate']:.2f}<{min_acceptance_rate}")
                subject_stats_df.loc[idx, 'excluded_low_acceptance_rate'] = True
            elif row['acceptance_rate'] > max_acceptance_rate:
                reasons.append(f"acceptance_rate={row['acceptance_rate']:.2f}>{max_acceptance_rate}")
                subject_stats_df.loc[idx, 'excluded_high_acceptance_rate'] = True

        if pd.notna(row['mean_rt']):
            if row['mean_rt'] < min_mean_rt:
                reasons.append(f"mean_rt={row['mean_rt']:.2f}<{min_mean_rt}")
                subject_stats_df.loc[idx, 'excluded_low_mean_rt'] = True
            elif row['mean_rt'] > max_mean_rt:
                reasons.append(f"mean_rt={row['mean_rt']:.2f}>{max_mean_rt}")
                subject_stats_df.loc[idx, 'excluded_high_mean_rt'] = True

        if (row['dataset'], row['sub']) in manually_excluded:
            reasons.append("manual_exclusion")
            subject_stats_df.loc[idx, 'excluded_manual'] = True

        if reasons:
            subject_stats_df.loc[idx, 'excluded'] = True
            subject_stats_df.loc[idx, 'exclusion_reason'] = '; '.join(reasons)

    # Filter behavioral data using a merge instead of row-wise apply
    excluded_subs = subject_stats_df.loc[subject_stats_df['excluded'], ['dataset', 'sub']].copy()
    excluded_subs['_exclude'] = True
    df = df.merge(excluded_subs, on=['dataset', 'sub'], how='left')
    df = df[df['_exclude'].isna()].drop(columns='_exclude').reset_index(drop=True)

    return df, subject_stats_df


def summarize_participant_exclusions(subject_stats_df):
    """
    Summarize participant-level exclusions by dataset and criterion.

    Criteria are evaluated independently (not cumulatively), so a participant
    excluded by multiple criteria is counted once under each criterion column,
    but only once in n_excluded_total.

    Parameters
    ----------
    subject_stats_df : pd.DataFrame
        Output of exclude_participants_by_criteria(), containing the
        per-criterion boolean columns and 'excluded'.

    Returns
    -------
    pd.DataFrame with columns:
        dataset, n_subjects,
        n_excluded_high_trial_exclusion_pct, n_excluded_low_acceptance_rate,
        n_excluded_high_acceptance_rate, n_excluded_low_mean_rt,
        n_excluded_high_mean_rt, n_excluded_manual,
        n_excluded_total (unique participants excluded, no double-counting)
    """
    criterion_cols = [
        'excluded_high_trial_exclusion_pct',
        'excluded_low_acceptance_rate',
        'excluded_high_acceptance_rate',
        'excluded_low_mean_rt',
        'excluded_high_mean_rt',
        'excluded_manual',
    ]

    rows = []
    for dataset, group in subject_stats_df.groupby('dataset'):
        row = {'dataset': dataset, 'n_subjects': len(group)}
        for col in criterion_cols:
            row[f'n_{col}'] = int(group[col].sum())
        row['n_excluded_total'] = int(group['excluded'].sum())
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_final_sample(subject_stats_df):
    """
    Summarize sample size and behavior statistics for included (non-excluded) participants.

    Parameters
    ----------
    subject_stats_df : pd.DataFrame
        Output of exclude_participants_by_criteria(), containing 'excluded',
        'n_valid_trials', 'acceptance_rate', and 'mean_rt' per subject.

    Returns
    -------
    pd.DataFrame with columns:
        dataset, n_subjects,
        mean_n_valid_trials, min_n_valid_trials, max_n_valid_trials,
        mean_acceptance_rate, sd_acceptance_rate,
        mean_rt, sd_rt
    """
    included = subject_stats_df[~subject_stats_df['excluded']]

    rows = []
    for dataset, group in included.groupby('dataset'):
        rows.append({
            'dataset': dataset,
            'n_subjects': len(group),
            'mean_n_valid_trials': group['n_valid_trials'].mean(),
            'min_n_valid_trials': group['n_valid_trials'].min(),
            'max_n_valid_trials': group['n_valid_trials'].max(),
            'mean_acceptance_rate': group['acceptance_rate'].mean(),
            'sd_acceptance_rate': group['acceptance_rate'].std(),
            'mean_rt': group['mean_rt'].mean(),
            'sd_rt': group['mean_rt'].std(),
        })

    return pd.DataFrame(rows)
