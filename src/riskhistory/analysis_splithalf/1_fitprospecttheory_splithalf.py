import pandas as pd

from riskhistory.analysis_acceptglme.fitprospecttheory import fit_all_subjects
from riskhistory.analysis_splithalf.utils import SPLITHALF_RESULTS_PTH, add_half_id
from riskhistory.data import load_behavioral_data

FITS = {
    "baseline":      {"choicehistory": 0.0},
    "choicehistory": None,
}


def fit_halves(data, fix, label):
    betas = []
    for half_id, df_half in data.groupby("half_id", sort=True):
        half_betas = fit_all_subjects(df_half, fix=fix, label=f"{label} (half {half_id})")
        half_betas.insert(2, "half_id", half_id)
        betas.append(half_betas)
    return pd.concat(betas, ignore_index=True)


def main():
    data = add_half_id(load_behavioral_data())
    SPLITHALF_RESULTS_PTH.mkdir(parents=True, exist_ok=True)

    for label, fix in FITS.items():
        betas = fit_halves(data, fix, label)
        out_file = SPLITHALF_RESULTS_PTH / f"1_prospecttheory_{label}_betas_splithalf.csv"
        betas.to_csv(out_file, index=False)
        print(f"Saved: {out_file.name}")


if __name__ == "__main__":
    main()
