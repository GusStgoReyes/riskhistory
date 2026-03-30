from riskhistory.config import (results_pth,
                                preprocessed_data_pth)

def save_betas_to_csv(results_df, exclusions, behavioral_data):
    # Save all of the extracted betas into the results
    results_dir = results_pth / "setup_pupilbetas"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "pupilbetas_lss_results.csv", index=False)

    # Merge behavioral and pupil exclusion criteria
    exclusions = exclusions[["dataset", "sub", "trial", "run", "exclude_startAligned"]]
    behavioral_data = behavioral_data[["dataset", "sub", "trial", "run", "exclude"]]
    exclude = exclusions.merge(behavioral_data, on=["dataset", "sub", "trial", "run"], how="left")
    exclude["final_exclude"] = exclude["exclude_startAligned"] | exclude["exclude"]

    # Query the trial_betas_names that contains "target"
    target_betas = results_df[results_df['trial_betas_names'].str.contains("target")]
    target_betas.loc[:, "trial"] = target_betas.loc[:, "target_trial"]
    target_betas = target_betas.drop(columns=["target_trial", "trial_betas_names"])
    target_betas = target_betas.reset_index(drop=True)
    target_betas = target_betas.rename(columns={"trial_betas_estimates": "pupil"})

    # zscore within run x subject (consistent with GLM being fit per run)
    target_betas["pupil"] = target_betas.groupby(["sub", "run"])["pupil"].transform(lambda x: (x - x.mean()) / x.std())

    target_betas = target_betas.merge(exclude[["dataset", "sub", "trial", "final_exclude"]], on=["dataset", "sub", "trial"], how="left")

    # Save target betas to preprocessed data folder
    preprocessed_data_pth.mkdir(parents=True, exist_ok=True)
    target_betas.to_csv(preprocessed_data_pth / "pupilbetas_lss_preprocessed.csv", index=False)


def save_glm_to_csv(results_df, exclusions):
    results_dir = results_pth / "setup_pupilbetas"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "pupilbetas_glm_results.csv", index=False)

    # Merge behavioral and pupil exclusion criteria
    exclusions = exclusions[["dataset", "sub", "trial", "run", "exclude_startAligned"]]

    target_betas = results_df[results_df['betas_names'].str.contains("trial")]
    target_betas["trial"] = (
        target_betas["betas_names"]
        .str.extract(r'trial-(\d+)')
        .astype(int)
    )
    target_betas = target_betas.drop(columns=["r2", "betas_names"])
    target_betas = target_betas.reset_index(drop=True)

    # zscore within run x subject (consistent with GLM being fit per run)
    target_betas["pupil"] = target_betas.groupby("sub")["betas_estimates"].transform(lambda x: (x - x.mean()) / x.std())
    target_betas["pupil_v2"] = target_betas.groupby(["sub", "run"])["betas_estimates"].transform(lambda x: (x - x.mean()) / x.std())

    target_betas = target_betas.merge(exclusions[["dataset", "sub", "trial", "exclude_startAligned"]], on=["dataset", "sub", "trial"], how="left")

    # Save target betas to preprocessed data folder
    preprocessed_data_pth.mkdir(parents=True, exist_ok=True)
    target_betas.to_csv(preprocessed_data_pth / "pupilbetas_glm_preprocessed.csv", index=False)
