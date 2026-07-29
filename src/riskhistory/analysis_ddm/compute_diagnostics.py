"""Per-model LOO, rhat, and ESS computation for a single HDDM fit.

Run one instance per model (e.g. as a SLURM array job) so each job holds
only one ~20 GB InferenceData file in memory at a time.
Results are saved as small intermediate files for compare_models.py to aggregate.
"""
import argparse
import pickle
from pathlib import Path

from .utils_data import load_data

import arviz as az
import numpy as np
import pandas as pd
import hddm

def idata_to_tidy(ds, model_id: str, stat_name: str) -> pd.DataFrame:
    rows = []
    for var in ds.data_vars:
        values = np.atleast_1d(ds[var].values).flatten()
        if len(values) == 1:
            rows.append({"model": model_id, "variable": var, stat_name: values[0]})
        else:
            for i, v in enumerate(values):
                rows.append({"model": model_id, "variable": f"{var}[{i}]", stat_name: v})
    return pd.DataFrame(rows)


def compute_diagnostics(model_id: str,
                        dataset: str,
                        results_pth: str,
                        pupil: bool,
                        data_pth: str,
                        pupil_pth: str,
                        pupil_col: str = "pupil",
                        ) -> None:
    # Must match model_hddm.py's save_name / utils_stats.py's idx_save suffix
    # exactly, so this locates the .nc file the corresponding fit wrote.
    idx_save = "" if pupil_col == "pupil" else f"_{pupil_col}"
    model_label = f"{model_id}{idx_save}"

    hddm_dir = Path(results_pth) / "hddm"
    output_dir = Path(results_pth) / "hddm_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_path = hddm_dir / f"{model_id}_{dataset}{idx_save}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    model_path = hddm_dir / f"{model_id}_{dataset}{idx_save}"
    m = hddm.load(model_path)
    print("p_outlier:", m.p_outlier, flush = True)

    print(f"Loading {nc_path} ...", flush=True)
    idata = az.from_netcdf(str(nc_path))
    
    print("Checking log_likelihood ...", flush=True)
    ll_da = idata.log_likelihood["log_lik"]
    ll = ll_da.values
    flat = ll.reshape(-1, ll.shape[-1])

    print("any nan:", np.isnan(flat).any(), "any inf:", np.isinf(flat).any(), flush = True)
    print("global min/max:", np.nanmin(flat), np.nanmax(flat), flush = True)
    var_t = flat.var(0); mean_t = flat.mean(0)
    print("p_waic:", var_t.sum(), flush = True)
    print("n var>1:", (var_t>1).sum(), " n>10:", (var_t>10).sum(), " n>100:", (var_t>100).sum(), flush = True)
    print("top 10 trial variances:", np.sort(var_t)[-10:], flush = True)
    print("avg per-trial mean loglik:", mean_t.mean(), " min:", mean_t.min(), flush = True)
    worst = np.argmax(var_t)
    print("worst trial sample:", flat[:20, worst], "min/max:", flat[:,worst].min(), flat[:,worst].max(), flush = True)

    # Fixing the log_likelihood to reflect p_outlier
    data = load_data(
        data_pth=data_pth,
        pupil_pth=pupil_pth,
        dataset_name=dataset,
        pupil=pupil,
        pupil_col=pupil_col,
    )
    rt_min, rt_max = data["rt"].min(), data["rt"].max()
    uniform_density = 1.0 / (rt_max - rt_min) / 2.0
    dens = np.exp(ll)
    ll_mixed = np.log((1 - m.p_outlier) * dens + m.p_outlier * uniform_density)

    # PSIS-LOO only fits a Generalized Pareto to each trial's *tail* (the
    # largest ~M draws, not the full posterior), so a trial can have nonzero
    # overall variance yet still have its top M values be exact ties (e.g.
    # the DDM likelihood saturates at a ceiling for most draws, with only a
    # minority differing) -- that's enough to trigger "All tail values are
    # the same" even though a whole-distribution variance check (like the
    # one this replaced) sees nothing wrong. Replicate PSIS's own tail-size
    # heuristic (reff=1 approximation) and check *that* slice specifically.
    flat_mixed = ll_mixed.reshape(-1, ll_mixed.shape[-1])
    n_samples = flat_mixed.shape[0]
    tail_size = max(5, int(min(0.2 * n_samples, 3 * np.sqrt(n_samples))))
    sorted_desc = np.sort(flat_mixed, axis=0)[::-1, :]
    tail_unique_counts = np.array([
        np.unique(sorted_desc[:tail_size, j]).size for j in range(sorted_desc.shape[1])
    ])
    degenerate = tail_unique_counts <= 1
    n_degenerate = int(degenerate.sum())
    print(f"Degenerate trials (constant tail, tail_size={tail_size}): "
          f"{n_degenerate} / {len(degenerate)}", flush=True)

    # .copy(data=...) preserves ll_da's chain/draw coordinates; assigning a
    # bare (dims, values) tuple instead drops them, which newer arviz_stats
    # versions require to be present (log_likelihood.chain.size in
    # _prepare_loo_inputs) -- causes AttributeError: 'DataArray' object has
    # no attribute 'chain'.
    ll_mixed_da = ll_da.copy(data=ll_mixed)
    if n_degenerate > 0:
        obs_dim = ll_da.dims[-1]
        keep_idx = np.where(~degenerate)[0]
        ll_mixed_da = ll_mixed_da.isel({obs_dim: keep_idx})
        print(f"Dropping {n_degenerate} degenerate trial(s) from LOO "
              f"(kept {len(keep_idx)}/{len(degenerate)})", flush=True)
    idata.log_likelihood["log_lik"] = ll_mixed_da

    print("  Computing LOO (pointwise) ...", flush=True)
    loo_result = az.loo(idata, pointwise=True)
    print("elpd_loo:", loo_result.elpd_loo, "p_loo:", loo_result.p_loo, flush = True)
    k = np.asarray(loo_result.pareto_k.values)
    print("n k>0.7:", (k>0.7).sum(), "max k:", k.max(), "new min ll:", ll_mixed.min(), flush = True)

    loo_path = output_dir / f"{model_id}_{dataset}{idx_save}_loo.pkl"
    with open(loo_path, "wb") as f:
        pickle.dump(loo_result, f)

    print("  Computing rhat ...", flush=True)
    idata_to_tidy(az.rhat(idata), model_label, "rhat").to_csv(
        output_dir / f"{model_id}_{dataset}{idx_save}_rhat.csv", index=False
    )

    print("  Computing ESS ...", flush=True)
    idata_to_tidy(az.ess(idata), model_label, "ess").to_csv(
        output_dir / f"{model_id}_{dataset}{idx_save}_ess.csv", index=False
    )

    print(f"Done. Results saved to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute LOO, rhat, and ESS for a single HDDM fit"
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="equalIndifference",
        choices=["equalIndifference", "sheng2020", "equalRange"],
    )
    parser.add_argument(
        "--data_pth",
        type=str,
        required=True,
        help="Path to behavioral CSV file",
    )
    parser.add_argument(
        "--pupil_pth",
        type=str,
        required=True,
        help="Path to pupil betas CSV file",
    )
    parser.add_argument(
        "--pupil_col",
        type=str,
        default="pupil",
        choices=["pupil", "pupil_residual"],
        help="Which pupil column the fit used as its 'pupil' regressor "
             "(must match how the model was fit, for filename lookup)",
    )
    parser.add_argument(
        "--results_pth", type=str, required=True, help="Path to results directory"
    )
    args = parser.parse_args()

    if "Pupil" in args.model_id:
        pupil = True
    else:
        pupil = False

    compute_diagnostics(
        model_id=args.model_id,
        dataset=args.dataset_name,
        results_pth=args.results_pth,
        pupil=pupil,
        data_pth=args.data_pth,
        pupil_pth=args.pupil_pth,
        pupil_col=args.pupil_col,
    )


if __name__ == "__main__":
    main()
