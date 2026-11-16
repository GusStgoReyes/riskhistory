from riskhistory.config import results_pth

SPLITHALF_RESULTS_PTH = results_pth / "analysis_splithalf"

HALF_BY_RUN = {
    "equalIndifference": {1: 1, 2: 1, 3: 2, 4: 2},
    "equalRange":        {1: 1, 2: 1, 3: 2, 4: 2},
    "sheng2020":         {1: 1, 2: 2},
}


def add_half_id(df):
    df = df.copy()
    df["half_id"] = [HALF_BY_RUN[ds][run] for ds, run in zip(df["dataset"], df["run"])]
    return df
