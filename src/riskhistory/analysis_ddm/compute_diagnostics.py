"""Per-model LOO, rhat, and ESS computation for a single HDDM fit.

Run one instance per model (e.g. as a SLURM array job) so each job holds only
one model's diagnostics data in memory at a time. Within a job, the
`log_likelihood` and `posterior` netCDF groups are loaded and processed one
at a time (never the full InferenceData at once) to bound peak memory for
40k-iteration chains. `p_outlier` is passed in rather than recovered via
`hddm.load()`, which would otherwise pull the entire (tens-of-GB) fitted
model into memory just for that one scalar.
Results are saved as small intermediate files for compare_models.py to aggregate.

Seems that the loglikelihood has to be injected the poutliers because version of the
pymc.
"""
import argparse
import pickle
from pathlib import Path

from .utils_data import load_data

import arviz as az
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp

def _nc_attrs(obj) -> dict:
    return {k: obj.getncattr(k) for k in obj.ncattrs()}


def print_nc_structure(nc_path: str) -> None:
    """Print every group/variable/attribute name in a netCDF file.

    Metadata only (shapes, dtypes, attrs) -- never reads variable data, so
    this is cheap and safe to run against arbitrarily large files.
    """
    with nc4.Dataset(nc_path, "r") as f:
        print(f"Global attrs: {_nc_attrs(f)}", flush=True)

        def walk(group, indent=""):
            print(f"{indent}Group: {group.path}", flush=True)
            print(f"{indent}  attrs: {_nc_attrs(group)}", flush=True)
            for vname, var in group.variables.items():
                print(
                    f"{indent}  var: {vname} shape={var.shape} dtype={var.dtype} "
                    f"attrs={_nc_attrs(var)}",
                    flush=True,
                )
            for subgroup in group.groups.values():
                walk(subgroup, indent + "  ")

        walk(f)


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
                        p_outlier: float = 0.50,
                        inspect_only: bool = False,
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

    if inspect_only:
        print_nc_structure(str(nc_path))
        return

    # p_outlier is a fixed scalar set at fit time (see model_hddm.py), passed
    # in rather than recovered via hddm.load() -- unpickling the full fitted
    # model just for this one number would pull tens of GB into memory.
    print("p_outlier:", p_outlier, flush=True)

    # Load only the netCDF groups each step actually needs, one at a time,
    # instead of the full InferenceData (posterior + log_likelihood +
    # sample_stats + ...) all at once.
    print(f"Loading log_likelihood group from {nc_path} ...", flush=True)
    # log_lik carries a `coordinates` attribute pointing at `node`/`subj_idx`/
    # `trial` (broadcast to the full (chain, draw, obs) shape -- an HDDM/pymc
    # storage artifact). xarray's CF decoding attaches those as
    # non-dimension coordinates on open, and a data_vars selection like
    # ds[["log_lik"]] does NOT drop existing coordinates -- so `node` (a
    # ~800M-element string array, far bigger than log_lik itself) still gets
    # pulled into memory even after "selecting only what we need" via
    # xarray. Reading log_lik directly through netCDF4 avoids xarray's CF
    # decoding entirely and never touches those arrays.
    with nc4.Dataset(str(nc_path), "r") as f:
        ll_grp = f.groups["log_likelihood"]
        ll_grp.set_auto_mask(False)
        ll_arr = np.asarray(ll_grp.variables["log_lik"][:])
        chain = np.asarray(ll_grp.variables["chain"][:])
        draw = np.asarray(ll_grp.variables["draw"][:])
        obs_id = np.asarray(ll_grp.variables["obs_id"][:])

    ll_ds = xr.DataArray(
        ll_arr,
        dims=("chain", "draw", "obs_id"),
        coords={"chain": chain, "draw": draw, "obs_id": obs_id},
        name="log_lik",
    ).to_dataset()

    print("Checking log_likelihood ...", flush=True)
    ll_da = ll_ds["log_lik"]
    ll = ll_da.values
    flat = ll.reshape(-1, ll.shape[-1])

    print("any nan:", np.isnan(flat).any(), "any inf:", np.isinf(flat).any(), flush = True)
    print("global min/max:", np.nanmin(flat), np.nanmax(flat), flush = True)

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

    # Theoretical floor of the loglikelihoods
    ll_mixed_floor = np.log(p_outlier * uniform_density)

    if np.any(ll < ll_mixed_floor):
        print("WARNING: Some log_likelihood values are below the theoretical floor "
              f"({ll_mixed_floor:.3f}). This may indicate numerical issues.", flush=True)

        log_terms = np.stack([ll, np.full_like(ll, np.log(uniform_density))])
        print("log_terms shape:", log_terms.shape, flush=True)
        weights = np.array([1 - p_outlier, p_outlier])
        b = weights.reshape((2,) + (1,) * (log_terms.ndim - 1))

        ll_mixed = ll #logsumexp(log_terms, b=b, axis=0)
    else:
        ll_mixed = ll

    ll_mixed_da = ll_da.copy(data=ll_mixed)
    ll_ds["log_lik"] = ll_mixed_da
    idata_ll = az.convert_to_datatree(ll_ds, group="log_likelihood")

    print("  Computing LOO (pointwise) ...", flush=True)
    # reff=1.0 (i.e. assume no autocorrelation) sidesteps az.loo's internal
    # requirement for a `posterior` group to estimate effective sample size --
    # loading the full 40k-draw posterior just for that isn't worth it here.
    # This only affects the reported SE's precision, not the elpd_loo point
    # estimate itself.
    loo_result = az.loo(idata_ll, pointwise=True, reff=1.0)
    print("elpd_loo:", loo_result.elpd, "p_loo:", loo_result.p, flush=True)

    k = np.asarray(loo_result.pareto_k.values)
    print("n k>0.7:", (k>0.7).sum(), "max k:", k.max(), "new min ll:", ll_mixed.min(), flush = True)

    loo_path = output_dir / f"{model_id}_{dataset}{idx_save}_loo.pkl"
    with open(loo_path, "wb") as f:
        pickle.dump(loo_result, f)

    del idata_ll, ll_ds, ll_da, ll, flat, ll_mixed, ll_mixed_da

    print(f"Loading posterior group from {nc_path} ...", flush=True)
    with xr.open_dataset(str(nc_path), group="posterior") as post_ds:
        post_ds.load()

    print("  Computing rhat ...", flush=True)
    idata_to_tidy(az.rhat(post_ds), model_label, "rhat").to_csv(
        output_dir / f"{model_id}_{dataset}{idx_save}_rhat.csv", index=False
    )

    print("  Computing ESS ...", flush=True)
    idata_to_tidy(az.ess(post_ds), model_label, "ess").to_csv(
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
    parser.add_argument(
        "--p_outlier",
        type=float,
        default=0.50,
        help="Outlier mixture probability the model was fit with "
             "(must match P_OUTLIER in sbatch/configs/general.sh / the "
             "--p_outlier passed to model_hddm.py for this fit)",
    )
    parser.add_argument(
        "--inspect_nc",
        action="store_true",
        help="Print every group/variable/attribute name in the .nc file "
             "(metadata only, no data read) and exit without computing "
             "diagnostics.",
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
        p_outlier=args.p_outlier,
        inspect_only=args.inspect_nc,
    )


if __name__ == "__main__":
    main()
