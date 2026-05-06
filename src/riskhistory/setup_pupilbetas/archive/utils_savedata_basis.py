from riskhistory.config import results_pth, preprocessed_data_pth


def save_glm_basis_to_csv(results_df, exclusions):
    results_dir = results_pth / "setup_pupilbetas"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "pupilbetas_glm_basis_results.csv", index=False)

    exclusions = exclusions[["dataset", "sub", "trial", "run", "exclude_startAligned"]]

    # Decision betas (one collapsed magnitude per trial)
    decision_betas = results_df[results_df["betas_names"].str.contains("decision")].copy()
    decision_betas["trial"] = (
        decision_betas["betas_names"]
        .str.extract(r"trial-(\d+)")
        .astype(int)
    )
    decision_betas = decision_betas.drop(columns=["r2", "betas_names"]).reset_index(drop=True)
    decision_betas["pupil"] = decision_betas.groupby(["sub", "run"])["betas_estimates"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Response betas (one collapsed magnitude per trial)
    response_betas = results_df[results_df["betas_names"].str.contains("response")].copy()
    response_betas["trial"] = (
        response_betas["betas_names"]
        .str.extract(r"trial-(\d+)")
        .astype(int)
    )
    response_betas = response_betas.drop(columns=["r2", "betas_names"]).reset_index(drop=True)
    response_betas["pupil"] = response_betas.groupby(["sub", "run"])["betas_estimates"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    target_betas = decision_betas.merge(
        response_betas,
        on=["sub", "run", "dataset", "trial"],
        suffixes=("_decision", "_response"),
    )
    target_betas = target_betas.merge(
        exclusions[["dataset", "sub", "trial", "exclude_startAligned"]],
        on=["dataset", "sub", "trial"],
        how="left",
    )

    preprocessed_data_pth.mkdir(parents=True, exist_ok=True)
    target_betas.to_csv(preprocessed_data_pth / "pupilbetas_glm_basis_preprocessed.csv", index=False)
