"""Data loading utilities for HDDM fitting."""
import hddm
import pandas as pd


def load_data(
    data_pth: str,
    pupil_pth: str,
    dataset_name: str,
    subj_IDs: list = None,
    pupil: bool = False,
) -> pd.DataFrame:
    """Load and preprocess behavioral data for HDDM fitting.

    Parameters
    ----------
    data_pth : str
        Path to the behavioral CSV file.
    pupil_pth : str
        Path to the pupil betas CSV file.
    dataset_name : str
        Name of the dataset to filter (e.g., 'equalIndifference', 'sheng2020').
    subj_IDs : list, optional
        List of subject IDs to include. If None, all subjects are included.
    pupil : bool, optional
        If True, include pupil data in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for HDDM fitting.
    """
    data = hddm.load_csv(data_pth)

    if pupil:
        pupil_data = hddm.load_csv(pupil_pth)
        data = data.drop(columns=["exclude"])
        data = pd.merge(data, pupil_data, on=["sub", "trial", "run", "dataset"], how="inner")
        data = data.rename(columns={"final_exclude": "exclude"})
    
    data = data[data["dataset"] == dataset_name]

    if subj_IDs is not None:
        data = data[data["sub"].isin(subj_IDs)]

    data = data.rename(columns={"accept": "response", "sub": "subj_idx"})
    data = data[~data["exclude"]]
    data = hddm.utils.flip_errors(data)

    return data
