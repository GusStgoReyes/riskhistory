"""Data loading utilities for HDDM fitting."""
import hddm
import pandas as pd


def load_data(
    data_pth: str,
    pupil_pth: str,
    dataset_name: str,
    subj_IDs: list = None,
    pupil: bool = False,
    pupil_col: str = "pupil",
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
    pupil_col : str, optional
        Which column in the pupil CSV to expose as "pupil" for the model
        formula -- e.g. "pupil" (raw GLM amplitude) or "pupil_residual"
        (residualized on RT/gain/loss/entropy/|delta EV|, see
        analysis_pupilbetas/2_pupilbetas_fullmodel.R). Ignored if pupil=False.

    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for HDDM fitting.
    """
    data = hddm.load_csv(data_pth)

    if pupil:
        pupil_data = hddm.load_csv(pupil_pth)
        # The old pupilbetas_glm_preprocessed.csv carries its own
        # exclude_startAligned flag; results/analysis_pupilbetas/
        # 2_pupilbetas_fullmodel_residuals.csv has no such column since both
        # behavioral and pupil-specific exclusions were already applied
        # upstream (analysis_pupilbetas/0_config.R) -- an inner merge alone
        # restricts to that already-excluded trial set.
        has_own_exclude = "exclude_startAligned" in pupil_data.columns
        if has_own_exclude:
            data = data.drop(columns=["exclude"])
        data = pd.merge(data, pupil_data, on=["sub", "trial", "run", "dataset"], how="inner")
        if has_own_exclude:
            data = data.rename(columns={"exclude_startAligned": "exclude"})
        if pupil_col != "pupil":
            data = data.drop(columns=["pupil"]).rename(columns={pupil_col: "pupil"})

    data = data[data["dataset"] == dataset_name]

    if subj_IDs is not None:
        data = data[data["sub"].isin(subj_IDs)]

    data = data.rename(columns={"accept": "response", "sub": "subj_idx"})
    data = data[~data["exclude"]]
    data = hddm.utils.flip_errors(data)

    return data
