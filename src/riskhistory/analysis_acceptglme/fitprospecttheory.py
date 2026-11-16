# =============================================================================
# Choice History Under Risk: Per-subject prospect-theory (loss aversion) fits
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from scipy.special import expit

from riskhistory.config import results_pth
from riskhistory.data import load_behavioral_data

PARAM_NAMES = ("bias", "theta", "lambda_", "choicehistory")

DEFAULT_BOUNDS = {
    "bias": (None, None),
    "theta": (1e-6, None),
    "lambda_": (1e-6, None),
    "choicehistory": (None, None),
}

DEFAULT_START = {"bias": 0.0, "theta": 1.0, "lambda_": 2.0, "choicehistory": 0.0}

# Basin hopping (global search) settings: random perturbations of the free
# parameters, each followed by a bounded L-BFGS-B local fit.
BH_NITER = 100
BH_STEPSIZE = 0.5
BH_SEED = 42


def _p_accept(params, gain, loss, prev_choice):
    bias, theta, lambda_, choicehistory = params

    # Subjective value
    subjective_value = 1/2*gain - 1/2*lambda_ * loss

    # Logistic choice rule
    sv_inv = np.where(subjective_value > 0, subjective_value, -np.abs(subjective_value) / lambda_)
    x = theta * (prev_choice*choicehistory + bias + sv_inv)

    # Probability
    eps = 1e-12
    return np.clip(expit(x), eps, 1 - eps)


def _neg_log_likelihood(params, gain, loss, accept, prev_choice):
    p_accept = _p_accept(params, gain, loss, prev_choice)

    return -np.sum(
        accept * np.log(p_accept) + (1 - accept) * np.log(1 - p_accept)
    )


def fit_lossaversion(
    gain,
    loss,
    accept,
    prev_choice,
    start=None,
    fix=None,
    bounds=None,
    niter=BH_NITER,
    stepsize=BH_STEPSIZE,
    seed=BH_SEED,
):
    gain = np.asarray(gain, dtype=float)
    loss = np.asarray(loss, dtype=float)
    accept = np.asarray(accept, dtype=float)
    prev_choice = np.asarray(prev_choice, dtype=float)

    fix = dict(fix or {})
    unknown = set(fix) - set(PARAM_NAMES)
    if unknown:
        raise ValueError(f"Unknown parameter(s) in fix: {sorted(unknown)}")

    start = {**DEFAULT_START, **(start or {})}
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}

    # Only the free parameters are searched; fixed ones are inserted back
    # before evaluating the likelihood, so basin hopping never perturbs them.
    free = [n for n in PARAM_NAMES if n not in fix]
    free_bounds = [bounds[n] for n in free]
    lower = np.array([-np.inf if lo is None else lo for lo, _ in free_bounds])
    upper = np.array([np.inf if hi is None else hi for _, hi in free_bounds])

    def full_params(x_free):
        values = {**fix, **dict(zip(free, x_free))}
        return [values[n] for n in PARAM_NAMES]

    def objective(x_free):
        return _neg_log_likelihood(full_params(x_free), gain, loss, accept, prev_choice)

    # Reject hops that land outside the bounds (e.g. theta or lambda <= 0)
    def within_bounds(x_new, **kwargs):
        return bool(np.all(x_new >= lower) and np.all(x_new <= upper))

    result = basinhopping(
        objective,
        x0=[start[n] for n in free],
        niter=niter,
        stepsize=stepsize,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": free_bounds},
        accept_test=within_bounds,
        rng=np.random.default_rng(seed),
    )
    best_x = full_params(result.x)

    # Proportion of trials whose choice the fitted model predicts correctly
    # (p_accept > 0.5 -> accept), counting both accepts and rejects.
    predicted_accept = (_p_accept(best_x, gain, loss, prev_choice) > 0.5)

    out = dict(zip(PARAM_NAMES, best_x))
    out["nll"] = float(result.fun)
    out["n_trials"] = int(gain.size)
    out["accuracy"] = float(np.mean(predicted_accept == (accept == 1)))
    # success of the local fit at the best basin found
    out["success"] = bool(result.lowest_optimization_result.success)
    return out


def fit_all_subjects(data, fix, label):
    """Fit the model per dataset x subject and return one row per subject."""
    rows = []
    for (dataset, sub), df_sub in data.groupby(["dataset", "sub"], sort=True):
        fit = fit_lossaversion(
            df_sub["gain"], df_sub["loss"], df_sub["accept"], df_sub["choice_prev1"],
            fix=fix,
        )
        rows.append({"dataset": dataset, "sub": sub, **fit})

    betas = pd.DataFrame(rows)
    n_ok = int(betas["success"].sum())
    print(f"{label}: {n_ok} of {len(betas)} subjects converged.")
    return betas


def main():
    data = load_behavioral_data()  # drops exclude == True trials
    print(f"Fitting {len(data)} trials from "
          f"{data.groupby(['dataset', 'sub']).ngroups} subjects.")

    out_pth = Path(results_pth) / "analysis_acceptglme"
    out_pth.mkdir(parents=True, exist_ok=True)

    # (A) Baseline: no choice history
    baseline_betas = fit_all_subjects(
        data, fix={"choicehistory": 0.0}, label="(A) baseline (choicehistory = 0)"
    )
    baseline_file = out_pth / "1_prospecttheory_baseline_betas.csv"
    baseline_betas.to_csv(baseline_file, index=False)
    print(f"Saved: {baseline_file.name}")

    # (B) Choice history: all parameters free
    choicehistory_betas = fit_all_subjects(
        data, fix=None, label="(B) choice history"
    )
    choicehistory_file = out_pth / "1_prospecttheory_choicehistory_betas.csv"
    choicehistory_betas.to_csv(choicehistory_file, index=False)
    print(f"Saved: {choicehistory_file.name}")


if __name__ == "__main__":
    main()
