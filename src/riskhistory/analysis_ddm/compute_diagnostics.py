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
                        ) -> None:
    hddm_dir = Path(results_pth) / "hddm"
    output_dir = Path(results_pth) / "hddm_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_path = hddm_dir / f"{model_id}_{dataset}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    model_path = hddm_dir / f"{model_id}_{dataset}"
    m = hddm.load(model_path)
    print("p_outlier:", m.p_outlier, flush = True)

    print(f"Loading {nc_path} ...", flush=True)
    idata = az.from_netcdf(str(nc_path))
    
    print("Checking log_likelihood ...", flush=True)
    ll = idata.log_likelihood["log_lik"].values
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
        pupil=pupil
    )
    rt_min, rt_max = data["rt"].min(), data["rt"].max()
    uniform_density = 1.0 / (rt_max - rt_min) / 2.0
    dens = np.exp(ll.values)
    ll_mixed = np.log((1 - m.p_outlier) * dens + m.p_outlier * uniform_density)
    idata.log_likelihood["log_lik"] = (ll.dims, ll_mixed)

    print("  Computing LOO (pointwise) ...", flush=True)
    loo_result = az.loo(idata, pointwise=True)
    print("elpd_loo:", loo_result.elpd_loo, "p_loo:", loo_result.p_loo, flush = True)
    k = np.asarray(loo_result.pareto_k.values)
    print("n k>0.7:", (k>0.7).sum(), "max k:", k.max(), "new min ll:", ll_mixed.min(), flush = True)

    loo_path = output_dir / f"{model_id}_{dataset}_loo.pkl"
    with open(loo_path, "wb") as f:
        pickle.dump(loo_result, f)

    print("  Computing rhat ...", flush=True)
    idata_to_tidy(az.rhat(idata), model_id, "rhat").to_csv(
        output_dir / f"{model_id}_{dataset}_rhat.csv", index=False
    )

    print("  Computing ESS ...", flush=True)
    idata_to_tidy(az.ess(idata), model_id, "ess").to_csv(
        output_dir / f"{model_id}_{dataset}_ess.csv", index=False
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
        pupil_pth=args.pupil_pth
    )


if __name__ == "__main__":
    main()
