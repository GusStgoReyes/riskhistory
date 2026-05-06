from riskhistory.config import (results_pth,
                                preprocessed_data_pth)

def save_glm_to_csv(results_df, exclusions):
    results_dir = results_pth / "setup_pupilbetas"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "pupilbetas_glm_results.csv", index=False)

    # Merge behavioral and pupil exclusion criteria
    exclusions = exclusions[["dataset", "sub", "trial", "run", "exclude_startAligned"]]

    # Decision betas (LS-A): one per trial, spanning stimulus onset to response
    target_betas = results_df[results_df['betas_names'].str.contains("decision")].copy()
    target_betas["trial"] = (
        target_betas["betas_names"]
        .str.extract(r'trial-(\d+)')
        .astype(int)
    )
    target_betas = target_betas.drop(columns=["r2", "betas_names"])
    target_betas = target_betas.reset_index(drop=True)
    target_betas["pupil"] = target_betas.groupby(["sub", "run"])["betas_estimates"].transform(lambda x: (x - x.mean()) / x.std())
    # Retain run-level alpha (same value per run, kept for provenance)
    if "alpha" not in target_betas.columns:
        target_betas["alpha"] = float("nan")

    # Merge with exclusions
    target_betas = target_betas.merge(exclusions[["dataset", "sub", "trial", "exclude_startAligned"]], on=["dataset", "sub", "trial"], how="left")

    # Save target betas to preprocessed data folder
    preprocessed_data_pth.mkdir(parents=True, exist_ok=True)
    target_betas.to_csv(preprocessed_data_pth / "pupilbetas_glm_preprocessed.csv", index=False)
