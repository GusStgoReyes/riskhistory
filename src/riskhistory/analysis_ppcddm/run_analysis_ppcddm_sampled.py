"""Posterior-predictive PPC-DDM simulation with sampled (not point-estimate)
parameters.

run_analysis_ppcddm.py simulates each (dataset, sub, model) once, using the
point-estimate parameters from the summarized hddm_stats/hddm_params CSVs.
This script instead draws NUM_PARAM_DRAWS parameter sets from the model's
actual posterior (see compute_diagnostics.py's save_posterior_npz), runs a
full simulation for each draw, and averages the resulting per-trial
statistics across draws -- a proper posterior-predictive summary that
propagates parameter uncertainty, rather than a single plug-in simulation.

The per-trial statistics computed within one draw (mean_rt, mean_accept,
pred_repeat -- see ddm_model.py's generate_data) are unchanged; only how the
parameters feeding each draw are chosen is new. The final output has the
same one-row-per-trial shape as run_analysis_ppcddm.py's, so
analysis_ddmppcbehavior's R code needs no changes.
"""
import argparse
import hashlib
import numpy as np
import pandas as pd

from riskhistory.analysis_ppcddm.model_factory import get_model
from riskhistory.analysis_ppcddm.utils import rename_hddm_param
from riskhistory.analysis_ppcddm.run_analysis_ppcddm import (
    MODELS,
    PUPIL_MODEL_MARKER,
    NUM_SAMPLES_PER_CONDITION,
    load_behavioral_data_pupilclean,
    generate_simulated_data,
    _build_job_manifest,
    _append_with_lock,
)
from riskhistory.config import behavioral_data_pth, results_pth

NUM_PARAM_DRAWS = 1000


# ---------------------------------------------------------------------------
# Posterior loading / draw selection
# ---------------------------------------------------------------------------

def load_posterior_npz(model_id: str, dataset: str) -> dict:
    """One model's posterior draws (compute_diagnostics.py's
    save_posterior_npz): "{raw_param}" and "{raw_param}_subj.{sub}" arrays
    of shape (n_chain, n_draw), keyed by HDDM's own raw param_name and the
    actual `sub` label (not a synthetic index), plus "_chain"/"_draw"
    coordinate arrays."""
    path = results_pth / "hddm_posteriors" / f"{model_id}_{dataset}_posterior.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run compute_diagnostics.py first")
    return dict(np.load(path))


def _draw_indices(model_id: str, dataset: str, n_chain: int, n_draw: int,
                  num_draws: int = NUM_PARAM_DRAWS) -> list[tuple[int, int]]:
    """num_draws distinct (chain_idx, draw_idx) pairs, sampled without
    replacement out of the n_chain * n_draw available posterior samples.

    Seeded only on (model_id, dataset) -- NOT sub -- so every subject's
    independent SLURM array task deterministically reproduces the identical
    draw sequence: iteration i is the same joint posterior sample for every
    subject of that model/dataset, preserving the between-subject
    correlation induced by shared group-level hyperparameters (sampling
    each subject's draws independently would break that). Uses a hash of
    the encoded string rather than Python's builtin hash(), which is
    randomized per-process (PYTHONHASHSEED) and would silently break that
    cross-process reproducibility.
    """
    digest = hashlib.sha256(f"{model_id}|{dataset}".encode()).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    n_total = n_chain * n_draw
    flat_idx = rng.choice(n_total, size=num_draws, replace=False)
    chain_idx, draw_idx = np.unravel_index(flat_idx, (n_chain, n_draw))
    return list(zip(chain_idx.tolist(), draw_idx.tolist()))


# ---------------------------------------------------------------------------
# Per-draw parameter setup
# ---------------------------------------------------------------------------

def setup_model_parameters_sampled(model_id: str, posterior: dict,
                                   chain_idx: int, draw_idx: int, sub: int):
    """One posterior draw's worth of parameters -- built the same way
    setup_model_parameters() combines group+subject CSV rows, except every
    value here comes from the single (chain_idx, draw_idx) posterior sample
    instead of a point estimate, and subject-specific values are looked up
    directly as f"{raw}_subj.{sub}" (HDDM's own per-subject node naming)
    rather than via a separate subj_params CSV. Falls back to the
    population-level draw when this subject has no subject-level node for a
    given parameter, mirroring setup_model_parameters()'s CSV-based
    fallback.
    """
    model = get_model(model_id)
    required = set(model.parameters_bounds.keys())

    # Only base (non-"_subj.") keys -- these are guaranteed present as a
    # fallback for every parameter, and _std hyperparameters (irrelevant to
    # set_parameters()) get filtered out below along with anything else the
    # model doesn't actually need.
    raw_names = {k for k in posterior if k not in ("_chain", "_draw") and "_subj." not in k}

    param_dict = {}
    for raw in raw_names:
        subj_key = f"{raw}_subj.{sub}"
        arr = posterior[subj_key] if subj_key in posterior else posterior[raw]
        param_dict[rename_hddm_param(raw)] = float(arr[chain_idx, draw_idx])

    missing = required - param_dict.keys()
    if missing:
        raise ValueError(
            f"{model_id}: posterior draw is missing required parameter(s) "
            f"{sorted(missing)} -- check compute_diagnostics.py's posterior "
            f"npz for this model/dataset."
        )
    # set_parameters() requires an exact key match -- drop anything the
    # model doesn't use (e.g. another mechanism's lag term, _std entries).
    param_dict = {k: v for k, v in param_dict.items() if k in required}

    # PyDDM Boundary is half of HDDM's a -- same halving as
    # setup_model_parameters() in run_analysis_ppcddm.py.
    for key in ("a", "a_Intercept", "a_pupil"):
        if key in param_dict:
            param_dict[key] = param_dict[key] / 2.0

    model.set_parameters(param_dict)
    return model


# ---------------------------------------------------------------------------
# Posterior-predictive simulation
# ---------------------------------------------------------------------------

def generate_simulated_data_sampled(model_id: str, dataset: str, sub: int,
                                    behavioral_data: pd.DataFrame, posterior: dict,
                                    num_draws: int = NUM_PARAM_DRAWS) -> pd.DataFrame:
    """num_draws independent DDM simulations, one per posterior parameter
    draw -- each internally averaged over NUM_SAMPLES_PER_CONDITION samples
    per trial via the unmodified generate_simulated_data() -- then averaged
    again across draws into the same per-trial columns
    generate_simulated_data() already produces (posterior-predictive mean,
    marginalizing over parameter uncertainty). Per-draw detail is discarded
    once averaged, so the output has the same one-row-per-trial shape as
    run_analysis_ppcddm.py's."""
    n_chain = posterior["_chain"].shape[0]
    n_draw = posterior["_draw"].shape[0]
    draws = _draw_indices(model_id, dataset, n_chain, n_draw, num_draws)

    per_draw = []
    for i, (chain_idx, draw_idx) in enumerate(draws):
        model = setup_model_parameters_sampled(model_id, posterior, chain_idx, draw_idx, sub)
        draw_df = generate_simulated_data(model, dataset, sub, behavioral_data,
                                          num_samples_per_condition=NUM_SAMPLES_PER_CONDITION)
        draw_df["param_draw"] = i
        per_draw.append(draw_df)

    long_df = pd.concat(per_draw, ignore_index=True)

    stat_cols = ["mean_rt", "mean_accept", "pred_repeat"]
    group_cols = [c for c in long_df.columns if c not in stat_cols + ["param_draw"]]
    return long_df.groupby(group_cols, as_index=False, dropna=False)[stat_cols].mean()


# ---------------------------------------------------------------------------
# Single-subject entry point (called by each array job)
# ---------------------------------------------------------------------------

def run_subject_sampled(dataset_name: str, sub: int, behavioral_data: pd.DataFrame,
                        behavioral_data_pupilclean: pd.DataFrame,
                        models: dict = MODELS, num_draws: int = NUM_PARAM_DRAWS) -> None:
    """Same per-(dataset, sub) shape as run_subject() in
    run_analysis_ppcddm.py, but posterior-predictive rather than
    point-estimate: loads that model's posterior npz and calls
    generate_simulated_data_sampled() instead of generate_simulated_data()
    directly."""
    dataset_behavioral_data = behavioral_data.query(f"dataset == '{dataset_name}' & sub == {sub}")
    dataset_behavioral_data_pupilclean = behavioral_data_pupilclean.query(
        f"dataset == '{dataset_name}' & sub == {sub}"
    )

    for model_name, model_id in models.items():
        is_pupil_model = PUPIL_MODEL_MARKER in model_id
        sub_data = dataset_behavioral_data_pupilclean if is_pupil_model else dataset_behavioral_data

        if sub_data.empty:
            population = "pupil-clean" if is_pupil_model else "behavioral"
            print(f"  [{dataset_name} / sub-{sub}] model: {model_name} -- "
                 f"no {population} trials for this subject, skipping")
            continue

        print(f"  [{dataset_name} / sub-{sub}] model: {model_name} "
             f"({num_draws} posterior draws)")

        posterior = load_posterior_npz(model_id, dataset_name)
        summary_df = generate_simulated_data_sampled(model_id, dataset_name, sub, sub_data,
                                                      posterior, num_draws=num_draws)

        out_path = results_pth / "analysis_ppcddm" / f"{model_id}_simulated_data_sampled.csv"
        _append_with_lock(out_path, summary_df)
        print(f"    Saved {len(summary_df)} rows -> {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPC-DDM posterior-predictive simulation (sampled parameters)"
    )
    parser.add_argument("--job_index", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID: index into (dataset, sub) manifest")
    parser.add_argument("--print_njobs", action="store_true",
                        help="Print total number of jobs and exit (for sbatch array sizing)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model_ids to simulate, e.g. 'HDDM_Base' "
                             "or 'HDDM_Base,HDDM_Base_Lag1-full' (default: all in MODELS)")
    parser.add_argument("--num_draws", type=int, default=NUM_PARAM_DRAWS,
                        help="Posterior parameter draws per subject/model (default: "
                             "%(default)s). Use a small number for a pilot timing run "
                             "before sizing the full array job's --time.")
    args = parser.parse_args()

    if args.models:
        selected_ids = [m.strip() for m in args.models.split(",")]
        models = {name: model_id for name, model_id in MODELS.items()
                 if model_id in selected_ids}
        unknown = set(selected_ids) - set(models.values())
        if unknown:
            raise ValueError(f"Unknown model_id(s) in --models: {sorted(unknown)}. "
                             f"Valid options: {sorted(MODELS.values())}")
    else:
        models = MODELS

    behavioral_data = (
        pd.read_csv(behavioral_data_pth)
        [["sub", "dataset", "trial", "gain_centered", "loss_centered", "choice_prev1", "accept", "repeat", "rt", "exclude"]]
        .query("exclude == False")
    )
    # Rename gain_centered/loss_centered (no _centered)
    behavioral_data["gain"] = behavioral_data["gain_centered"]
    behavioral_data["loss"] = behavioral_data["loss_centered"]

    behavioral_data_pupilclean = load_behavioral_data_pupilclean()

    # Union of both populations' (dataset, sub) pairs -- see
    # run_analysis_ppcddm.py's main() for why this is a union, not an
    # assumed subset.
    jobs = sorted(set(_build_job_manifest(behavioral_data))
                 | set(_build_job_manifest(behavioral_data_pupilclean)))

    if args.print_njobs:
        print(len(jobs))
        raise SystemExit(0)

    if args.job_index is not None:
        if args.job_index >= len(jobs):
            print(f"job_index {args.job_index} >= n_jobs {len(jobs)}, nothing to do.")
            raise SystemExit(0)
        jobs = [jobs[args.job_index]]

    for dataset_name, sub in jobs:
        print(f"\nProcessing dataset={dataset_name}, sub={sub}")
        run_subject_sampled(dataset_name, sub, behavioral_data, behavioral_data_pupilclean,
                            models, num_draws=args.num_draws)

    print("\nDone.")
