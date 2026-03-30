# =============================================================================
# Choice History Under Risk: Behavioral Data Preparation Functions
# =============================================================================
# Columns that should be present in the final dataframe:
# 'sub': subject identifier
# 'trial': trial number
# 'trial_run': trial number within the run
# 'accept': choice made by the subject (0 or 1)
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

def fix_sub_56(df):
    """
    For subject 56 of NARPS_equalRange dataset, flip the responses and remap them.
    """
    # Mapping for correct interpretation
    int_to_label = {
        1.0: "strongly_reject",
        2.0: "weakly_reject",
        3.0: "weakly_accept",
        4.0: "strongly_accept"
    }

    mask = df["sub"] == 56
    if mask.any():
        df.loc[mask, "response_int"] = 5 - df.loc[mask, "response_int"]

        # Recompute accept: 0 if reject, 1 if accept
        df.loc[mask, "accept"] = df.loc[mask, "response_int"].isin([3.0, 4.0]).astype(int)

        # Update participant_response
        df.loc[mask, "participant_response"] = df.loc[mask, "response_int"].map(int_to_label)
    
    return df

def response_to_int(response_int, 
                    RT, 
                    min_rt = 0.2,
                    max_rt = 3.8,
                    ordinal = False):
    """
    Convert response to integer code, setting to NaN if RT is below min_rt.
    If ordinal is True, return 1-4 scale, else return binary accept/reject.
    """
    if RT < min_rt or RT > max_rt:
        return np.nan
    else:
        if ordinal:
            if response_int in [0, 1, 2, 3, 4]:
                return response_int
            else:
                return np.nan
        else:
            if response_int == 1 or response_int == 2:
                return 0
            elif response_int == 3 or response_int == 4:
                return 1
            else:
                return np.nan
    
def new_row_trial(group, 
                  trial,
                  stim):
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
    # Iterate through StimulusName codes (ordered by time)
    for code in group["StimulusName"]:
        # If code is <= 100, then its an indicator for the trial number
        if code <= 100:
            if trial is not None:
                # Two trial stims in a row!
                rows.append(new_row_trial(group, trial, stim = None))

            trial = code
        # If code > 100, then its a stimulus code (which contains the Gain and Loss information)
        elif code > 100:
            if stim is not None:
                # Two stim codes in a row!
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

def create_choice_columns(df, 
                          button_column = 'response_int', 
                          N_max = 6):
    """
    Create choice and choice history columns in the dataframe.
    """
    # Create choice column
    df['choice'] = df['accept'].replace({0: -1, 1: 1})

    # Create choice_prev[N] columns
    for N in range(1, N_max + 1):
        df[f'choice_prev{N}'] = df.groupby(['sub', 'run'])['choice'].shift(N)
    
    # Replace nan with 0
    df[[f'choice_prev{N}' for N in range(1, N_max + 1)]] = df[[f'choice_prev{N}' for N in range(1, N_max + 1)]].fillna(0)

    # Create switch_button column
    df['switch_button'] = (df.groupby(['sub', 'run'])[button_column]
                            .transform(lambda x: np.where(
                                x.isna() | x.shift(1).isna(),  # if current or previous is NaN
                                np.nan,                        # output NaN
                                (x != x.shift(1)).astype(int)  # else 1 if switched, 0 if not
                            )))
    
    # Create repeat columns
    df['repeat'] = (df.groupby(['sub', 'run'])["accept"]
                            .transform(lambda x: np.where(
                                x.isna() | x.shift(1).isna(),  # if current or previous is NaN
                                np.nan,                        # output NaN
                                (x == x.shift(1)).astype(int)  # else 1 if same, 0 if not
                            )))

    # Create switch column (opposite of repeat)
    df['switch'] = df['repeat'].apply(lambda x: np.nan if pd.isna(x) else (0 if x == 1 else 1))
    
    return df

def calculate_streak(group):
    """Calculate streak: how many trials back to find the same choice."""
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
    # streak_on_switch: log(streak) when switch==1, 0 otherwise
    # streak_on_repeat: log(streak) when repeat==1, 0 otherwise
    streaks = df.groupby(['sub', 'run'], group_keys=False).apply(calculate_streak)
    df["streak"] = streaks
    
    df['logstreak_on_switch'] = np.where(df['switch'] == 1, 
                                       np.log(df['streak'] +10e-10), 
                                       0)
    
    df['logstreak_on_repeat'] = np.where(df['repeat'] == 1, 
                                       np.log(df['streak'] +10e-10), 
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

def prepare_narps_behavioral_data(df, 
                                  min_rt = 0.2, 
                                  max_rt = 3.8,
                                  N_max = 6):
    """
    Prepare NARPS behavioral data for analysis.
    """
    # Fix subject 56's responses
    df = fix_sub_56(df)

    # Add trial and trial_run columns
    df['trial'] = np.arange(0, len(df)) % 256
    df['trial'] += 1
    df['trial_run'] = np.arange(0, len(df)) % 64
    df['trial_run'] += 1

    # Rename RT and condition columns
    df = df.rename(columns={
        'RT' : 'rt',
        'condition' : 'dataset',
    })

    # Fix the accept column to have correct NaN for too fast or too slow responses
    df['accept'] = df[['response_int', 'rt']].apply(lambda row: response_to_int(row['response_int'], 
                                                                                row['rt'], 
                                                                                min_rt,
                                                                                max_rt,
                                                                                ordinal=False), 
                                                                                axis=1)
    
    # Fix the response_int column to have correct NaN for too fast or too slow responses
    df['response_int'] = df[['response_int', 'rt']].apply(lambda row: response_to_int(row['response_int'], 
                                                                                      row['rt'], 
                                                                                      min_rt, 
                                                                                      max_rt,
                                                                                      ordinal=True), 
                                                                                      axis=1)

    # Create choice and history columns
    df = create_choice_columns(df, 
                               button_column='response_int', 
                               N_max = N_max)

    # Add centered gain and loss columns
    df = add_centered_columns(df, dataset_column='dataset')

    # Add the prev_gain and prev_loss columns
    df['prev_gain'] = df.groupby(['sub', 'run'])['gain'].shift(1)
    df['prev_loss'] = df.groupby(['sub', 'run'])['loss'].shift(1)
    df['prev_gain_centered'] = df.groupby(['sub', 'run'])['gain_centered'].shift(1)
    df['prev_loss_centered'] = df.groupby(['sub', 'run'])['loss_centered'].shift(1)

    # Add log streak columns
    df = add_logstreak_columns(df)
    
    # Select and order final columns
    df = df[['sub', 'trial', 'trial_run', 'run', 'accept', 'gain', 'loss', 
            'gain_centered', 'loss_centered', 'prev_gain_centered', 'prev_loss_centered',
             'rt', 'dataset', 'choice'] +
            [f'choice_prev{N}' for N in range(1, 7)] +
            ['switch_button', 'repeat', 'switch', 'prev_gain', 'prev_loss',
             'streak', 'logstreak_on_switch', 'logstreak_on_repeat']]
    
    return df
        
def prepare_sheng2020_behavioral_data(df_response, 
                                      df_pupil, 
                                      min_rt = 0.2, 
                                      max_rt = 3.8,
                                      N_max = 6):
    """
    Prepare Sheng2020 behavioral data for analysis.
        - df_response: DataFrame containing the response data (trials are not ordered)
        - df_pupil: DataFrame containing the pupil data (used to get trial order)
    """
    # Keep only necessary columns and drop duplicates
    df_pupil = df_pupil[["Subject", "Run", "StimulusName"]]
    df_pupil = df_pupil.drop_duplicates(subset=["Subject", "Run", "StimulusName"]).reset_index(drop = True)

    # Apply per Subject/Run group to get ordered trial mapping
    trial_df = (
        df_pupil.groupby(["Subject", "Run"])[["Subject", "Run", "StimulusName"]]
        .apply(get_trial_ordering)
        .reset_index(drop=True)
    )

    # Add the gain and loss columns based on the StimulusCode
    trial_df = add_gain_loss(trial_df)
    trial_df = trial_df.drop("StimulusCode", axis=1)

    # Add the responses to the trial ordered dataframe
    df = trial_df.merge(df_response, on=["Subject", "Run", "Gain", "Loss"], how="left")

    # Convert the RT to seconds
    df["RT"] = df["RT"] / 1000

    # Add the trial and trial_run columns
    df["trial_run"] = df["trial"] + 1
    df["trial"] = df["trial"] + 1 + 100 * (df["Run"] - 1)

    # Rename the columns
    df = df.rename(columns={"Subject" : "sub", 
                            "Run" : "run", 
                            "Gain" : "gain", 
                            "Loss" : "loss", 
                            "RT" : "rt", 
                            "Response" : "accept"})
    
    # Add the dataset column
    df['dataset'] = 'sheng2020'

    # Fix the accept column to have correct NaN for too fast responses
    df['accept'] = df[['accept', 'rt']].apply(lambda row: response_to_int(row['accept'], 
                                                                          row['rt'], 
                                                                          min_rt, 
                                                                          max_rt, 
                                                                          ordinal=True), 
                                                                          axis=1)
    
    # Create choice and history columns
    df = create_choice_columns(df, 
                               button_column='accept',
                               N_max = N_max)

     # Add centered gain and loss columns
    df = add_centered_columns(df, dataset_column='dataset')

    # Add the prev_gain and prev_loss columns
    df['prev_gain'] = df.groupby(['sub', 'run'])['gain'].shift(1)
    df['prev_loss'] = df.groupby(['sub', 'run'])['loss'].shift(1)
    df['prev_gain_centered'] = df.groupby(['sub', 'run'])['gain_centered'].shift(1)
    df['prev_loss_centered'] = df.groupby(['sub', 'run'])['loss_centered'].shift(1)

    # Add log streak columns
    df = add_logstreak_columns(df)

    # Select and order final columns
    df = df[['sub', 'trial', 'trial_run', 'run', 'accept', 'gain', 'loss', 
             'gain_centered', 'loss_centered', 'prev_gain_centered', 'prev_loss_centered',
             'rt', 'dataset', 'choice'] +
            [f'choice_prev{N}' for N in range(1, 7)] +
            ['switch_button', 'repeat', 'switch', 'prev_gain', 'prev_loss',
             'streak', 'logstreak_on_switch', 'logstreak_on_repeat']]
    
    return df

def add_exclude_trials(df, min_rt = 0.2, max_rt = 3.8, min_prev_valid_trials = 2):
    # Initialize exclude column to False
    df['exclude'] = False

    # Exclude trials based on reaction time
    df.loc[(df['rt'] < min_rt) | (df['rt'] > max_rt), 'exclude'] = True

    # Exclude trials based on previous valid trials
    choice_histories = []
    for N in range(1, min_prev_valid_trials + 1):
        choice_histories.append(df[f'choice_prev{N}'] == 0)
    
    df.loc[np.logical_or.reduce(choice_histories), 'exclude'] = True

    # Exclude trials without a gain/loss value
    df.loc[df['gain'].isna() | df['loss'].isna(), 'exclude'] = True

    return df

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

        # Compute stats on valid (non-excluded) trials only
        valid_trials = group[~group['exclude']]

        # Count missing responses (NaN in accept column)
        n_missing = valid_trials['accept'].isna().sum()

        # Acceptance rate (excluding NaN responses)
        valid_responses = valid_trials['accept'].dropna()
        acceptance_rate = valid_responses.mean() if len(valid_responses) > 0 else np.nan

        # RT statistics (on valid trials with valid responses)
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
        Updated stats with 'excluded' and 'exclusion_reason' columns.
    """
    if manually_excluded is None:
        manually_excluded = []

    # Initialize exclusion columns
    subject_stats_df = subject_stats_df.copy()
    subject_stats_df['excluded'] = False
    subject_stats_df['exclusion_reason'] = ''

    def add_reason(row, reason):
        if row['exclusion_reason']:
            return row['exclusion_reason'] + '; ' + reason
        return reason

    # Check each criterion
    for idx, row in subject_stats_df.iterrows():
        reasons = []

        # Too many excluded trials
        if row['excluded_trials_pct'] > max_excluded_trials_pct:
            reasons.append(f"excluded_trials_pct={row['excluded_trials_pct']:.2f}")

        # Acceptance rate too low or too high
        if pd.notna(row['acceptance_rate']):
            if row['acceptance_rate'] < min_acceptance_rate:
                reasons.append(f"acceptance_rate={row['acceptance_rate']:.2f}<{min_acceptance_rate}")
            elif row['acceptance_rate'] > max_acceptance_rate:
                reasons.append(f"acceptance_rate={row['acceptance_rate']:.2f}>{max_acceptance_rate}")

        # Mean RT too fast or too slow
        if pd.notna(row['mean_rt']):
            if row['mean_rt'] < min_mean_rt:
                reasons.append(f"mean_rt={row['mean_rt']:.2f}<{min_mean_rt}")
            elif row['mean_rt'] > max_mean_rt:
                reasons.append(f"mean_rt={row['mean_rt']:.2f}>{max_mean_rt}")

        # Manual exclusion
        if (row['dataset'], row['sub']) in manually_excluded:
            reasons.append("manual_exclusion")

        if reasons:
            subject_stats_df.loc[idx, 'excluded'] = True
            subject_stats_df.loc[idx, 'exclusion_reason'] = '; '.join(reasons)

    # Get list of excluded participants (dataset, sub pairs)
    excluded_mask = subject_stats_df['excluded']
    excluded_pairs = set(
        zip(
            subject_stats_df.loc[excluded_mask, 'dataset'],
            subject_stats_df.loc[excluded_mask, 'sub']
        )
    )

    # Filter behavioral data to remove excluded participants
    df = df[
        ~df.apply(lambda row: (row['dataset'], row['sub']) in excluded_pairs, axis=1)
    ].reset_index(drop=True)

    return df, subject_stats_df