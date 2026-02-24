"""Data loading utilities for HDDM fitting."""
import hddm
import pandas as pd


def load_data(
    data_pth: str,
    dataset_name: str,
    subj_IDs: list = None,
) -> pd.DataFrame:
    """Load and preprocess behavioral data for HDDM fitting.

    Parameters
    ----------
    data_pth : str
        Path to the behavioral CSV file.
    dataset_name : str
        Name of the dataset to filter (e.g., 'equalIndifference', 'sheng2020').
    subj_IDs : list, optional
        List of subject IDs to include. If None, all subjects are included.

    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for HDDM fitting.
    """
    data = hddm.load_csv(data_pth)
    data = data[data["dataset"] == dataset_name]

    if subj_IDs is not None:
        data = data[data["sub"].isin(subj_IDs)]

    data = data.rename(columns={"accept": "response", "sub": "subj_idx"})
    data = data[~data["exclude"]]
    data = hddm.utils.flip_errors(data)

    return data
