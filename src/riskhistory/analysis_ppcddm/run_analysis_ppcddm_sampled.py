"""Posterior-predictive PPC-DDM simulation with sampled (not point-estimate)
parameters.
"""
import argparse
import hashlib
from pathlib import Path
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

NUM_PARAM_DRAWS = 1
DEFAULT_DDM_RESULTS_PTH = results_pth / "analysis_ddm_final"


# ---------------------------------------------------------------------------
# Posterior loading / draw selection
# ---------------------------------------------------------------------------

def load_posterior_npz(model_id: str, dataset: str, ddm_results_pth: Path) -> dict:
    """Posterior draws saved by compute_diagnostics.py: "{param}" and
    "{param}_subj.{sub}" arrays of shape (n_chain, n_draw), plus "_chain"/"_draw"."""
    path = Path(ddm_results_pth) / "hddm_posteriors" / f"{model_id}_{dataset}_posterior.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run compute_diagnostics.py first")
    return dict(np.load(path))


def _draw_indices(model_id: str, dataset: str, n_chain: int, n_draw: int,
                  num_draws: int = NUM_PARAM_DRAWS) -> list[tuple[int, int]]:
    """num_draws distinct (chain_idx, draw_idx) pairs, sampled without replacement.

    Seeded on (model_id, dataset) only, so every subject's array task uses the
    same draws and the group-level correlation between subjects is preserved.
    sha256 rather than hash(), which is randomized per process.
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
                                   draw: tuple[int, int] | None, sub: int):
    """Same as setup_model_parameters(), but from one posterior draw
    (chain_idx, draw_idx), or each parameter's posterior median if draw is
    None: the subject-level value if it exists, else the group-level one."""
    model = get_model(model_id)
    required = set(model.parameters_bounds.keys())

    raw_names = {k for k in posterior if k not in ("_chain", "_draw") and "_subj." not in k}

    param_dict = {}
    for raw in raw_names:
        subj_key = f"{raw}_subj.{sub}"
        arr = posterior[subj_key] if subj_key in posterior else posterior[raw]
        value = np.median(arr) if draw is None else arr[draw]
        param_dict[rename_hddm_param(raw)] = float(value)

    missing = required - param_dict.keys()
    if missing:
        raise ValueError(
            f"{model_id}: posterior draw is missing required parameter(s) "
            f"{sorted(missing)} -- check compute_diagnostics.py's posterior "
            f"npz for this model/dataset."
        )
    # set_parameters() needs an exact key match (drops _std etc.)
    param_dict = {k: v for k, v in param_dict.items() if k in required}

    # PyDDM's boundary is half of HDDM's a
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
    """Simulate once per posterior draw, then average across draws: one row per
    trial, same columns as generate_simulated_data(). With num_draws == 1,
    simulates once from the posterior medians instead of a random draw."""
    if num_draws == 1:
        draws = [None]
    else:
        n_chain = posterior["_chain"].shape[0]
        n_draw = posterior["_draw"].shape[0]
        draws = _draw_indices(model_id, dataset, n_chain, n_draw, num_draws)

    per_draw = []
    for i, draw in enumerate(draws):
        model = setup_model_parameters_sampled(model_id, posterior, draw, sub)
        draw_df = generate_simulated_data(model, dataset, sub, behavioral_data,
                                          num_samples_per_condition=NUM_SAMPLES_PER_CONDITION)
        draw_df["param_draw"] = i
        per_draw.append(draw_df)

    long_df = pd.concat(per_draw, ignore_index=True)

    stat_cols = ["mean_rt", "mean_accept", "pred_repeat"]
    group_cols = [c for c in long_df.columns if c not in stat_cols + ["param_draw"]]
    return long_df.groupby(group_cols, as_index=False, dropna=False)[stat_cols].mean()


# ---------------------------------------------------------------------------
# Job manifest: lists of (dataset, sub, model_id) triples, one list per job
# ---------------------------------------------------------------------------
# By default one job per (dataset, sub, model): at 1000 draws a single
# model/sub takes ~6.5 h. With per_subject (e.g. for median runs, ~25 s per
# model) one job per (dataset, sub) runs every selected model.

def _build_job_manifest_per_model(behavioral_data: pd.DataFrame,
                                  behavioral_data_pupilclean: pd.DataFrame,
                                  models: dict,
                                  per_subject: bool = False) -> list[list[tuple[str, int, str]]]:
    """Sorted jobs, each a list of (dataset, sub, model_id) triples; index =
    SLURM_ARRAY_TASK_ID. Subjects with no trials in a model's population are
    skipped at run time."""
    dataset_sub_pairs = sorted(set(_build_job_manifest(behavioral_data))
                               | set(_build_job_manifest(behavioral_data_pupilclean)))
    jobs = [[(ds, sub, model_id) for model_id in models.values()]
            for ds, sub in dataset_sub_pairs]
    if per_subject:
        return jobs
    return [[triple] for job in jobs for triple in job]


# ---------------------------------------------------------------------------
# Single (dataset, sub, model) entry point (called by each array job)
# ---------------------------------------------------------------------------

def run_subject_model_sampled(dataset_name: str, sub: int, model_id: str,
                              behavioral_data: pd.DataFrame,
                              behavioral_data_pupilclean: pd.DataFrame,
                              ddm_results_pth: Path,
                              num_draws: int = NUM_PARAM_DRAWS) -> None:
    """Simulate one (dataset, sub, model_id) and append it to that model's CSV."""
    is_pupil_model = PUPIL_MODEL_MARKER in model_id
    data = behavioral_data_pupilclean if is_pupil_model else behavioral_data
    sub_data = data.query(f"dataset == '{dataset_name}' & sub == {sub}")

    if sub_data.empty:
        population = "pupil-clean" if is_pupil_model else "behavioral"
        print(f"  [{dataset_name} / sub-{sub}] model: {model_id} -- "
             f"no {population} trials for this subject, skipping")
        return

    print(f"  [{dataset_name} / sub-{sub}] model: {model_id} "
         f"({num_draws} posterior draws)")

    posterior = load_posterior_npz(model_id, dataset_name, ddm_results_pth)
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
                        help="SLURM_ARRAY_TASK_ID: index into the job manifest")
    parser.add_argument("--print_njobs", action="store_true",
                        help="Print total number of jobs and exit (for sbatch array sizing)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model_ids to simulate, e.g. 'HDDM_Base' "
                             "or 'HDDM_Base,HDDM_Base_Lag1-full' (default: all in MODELS)")
    parser.add_argument("--num_draws", type=int, default=NUM_PARAM_DRAWS,
                        help="Posterior parameter draws per subject/model (default: "
                             "%(default)s); 1 simulates from the posterior medians instead.")
    parser.add_argument("--per_subject", action="store_true",
                        help="One job per (dataset, sub) running every selected model, "
                             "instead of one job per (dataset, sub, model)")
    parser.add_argument("--results_pth", type=Path, default=DEFAULT_DDM_RESULTS_PTH,
                        help="HDDM results folder containing hddm_posteriors/ "
                             "(default: %(default)s)")
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

    jobs = _build_job_manifest_per_model(behavioral_data, behavioral_data_pupilclean, models,
                                         per_subject=args.per_subject)

    if args.print_njobs:
        print(len(jobs))
        raise SystemExit(0)

    if args.job_index is not None:
        if args.job_index >= len(jobs):
            print(f"job_index {args.job_index} >= n_jobs {len(jobs)}, nothing to do.")
            raise SystemExit(0)
        jobs = [jobs[args.job_index]]

    for dataset_name, sub, model_id in (triple for job in jobs for triple in job):
        print(f"\nProcessing dataset={dataset_name}, sub={sub}, model={model_id}")
        run_subject_model_sampled(dataset_name, sub, model_id, behavioral_data,
                                  behavioral_data_pupilclean,
                                  ddm_results_pth=args.results_pth,
                                  num_draws=args.num_draws)

    print("\nDone.")
