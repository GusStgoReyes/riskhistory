"""Per-model LOO, rhat, and ESS computation for a single HDDM fit.

Run one instance per model (e.g. as a SLURM array job) so each job holds only
one model's diagnostics data in memory at a time. Within a job, the
`log_likelihood` and `posterior` netCDF groups are loaded and processed one
at a time (never the full InferenceData at once) to bound peak memory for
40k-iteration chains. `p_outlier` is passed in. 
Results are saved as small intermediate files for compare_models.py to aggregate.

Inherent problems with docker-hddm loglikelihoods:
(1) sample() calls to_infdata() which calls get_pointwise_loglike() 
https://github.com/panwanke/kabuki/blob/a2302364b8b76c36c10e386520cea6917d6c0565/kabuki/hierarchical.py#L1204
(2) get_pointwise_loglike() calls pointwise_like_gen() which calls _pointwise_like_generate() which calls hddm.wfpt.pdf_array()
https://github.com/panwanke/kabuki/blob/a2302364b8b76c36c10e386520cea6917d6c0565/kabuki/analyze.py
(3) pdf_array() function sets w_outlier = 0 | only returns log(y * (1 - p_outlier))
https://github.com/panwanke/hddm/blob/4205656814327fb709421e3f2287ba13c58e00e2/src/wfpt.pyx
(4) w_outlier is automatically set to 0.1 at fitting time, need to add these amounts to the loglikelihoods

"""
import argparse
import pickle
from pathlib import Path

import arviz as az
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

# HDDM's own outlier-mixture density (hddm/likelihoods.py, hddm_regression.py,
# etc. -- hardcoded default across every model class), NOT derived from the
# behavioral data's RT range. See module docstring: the log_lik saved to the
# .nc file is computed with this term dropped (w_outlier=0 at extraction time
# vs. 0.1 at fit time), so it must be added back in by hand.
W_OUTLIER = 0.1

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


def save_posterior_npz(
    post_ds, model_id: str, dataset: str, idx_save: str, results_pth: str,
    dtype=np.float32,
) -> Path:
    """Save an already-loaded posterior Dataset to a compressed .npz.

    Same output format as extract_posteriors.py, but takes the Dataset
    directly instead of reopening the .nc file -- compute_diagnostics.py
    already has it loaded for rhat/ess.
    """
    arrays = {var: da.values.astype(dtype) for var, da in post_ds.data_vars.items()}
    arrays["_chain"] = post_ds["chain"].values
    arrays["_draw"] = post_ds["draw"].values

    output_dir = Path(results_pth) / "hddm_posteriors"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_id}_{dataset}{idx_save}_posterior.npz"

    np.savez_compressed(out_path, **arrays)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {len(arrays) - 2} parameters to {out_path} ({size_mb:.1f} MB)", flush=True)
    return out_path


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
                        pupil_col: str = "pupil",
                        p_outlier: float = 0.05,
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

    # Correct the log_likelihood for the missing outlier-mixture term (see
    # module docstring): the saved log_lik is log(y * (1 - p_outlier)), so
    # log(exp(log_lik) + p_outlier * W_OUTLIER) recovers
    # log(y * (1 - p_outlier) + w_outlier * p_outlier), the actual mixture
    # density HDDM used at fit time. np.logaddexp is the numerically stable
    # form of this and correctly no-ops when p_outlier == 0 (log(0) = -inf
    # contributes nothing).
    outlier_log_term = np.log(p_outlier * W_OUTLIER) if p_outlier > 0 else -np.inf
    n_below = np.sum(ll < outlier_log_term)
    print(f"n log_lik values below the true mixture floor ({outlier_log_term:.3f}): "
          f"{n_below} (will be corrected)", flush=True)

    ll_mixed = ll # np.logaddexp(ll, outlier_log_term)

    # Per-trial diagnostics for the outlier-dominated observations that break
    # PSIS's Pareto-tail fit ("All tail values are the same"): once the raw
    # log-likelihood is below the floor on (nearly) every draw, the corrected
    # value collapses to ~outlier_log_term regardless of the parameters, so
    # its variance across draws goes to ~0 -- exactly what breaks the fit.
    flat_mixed = ll_mixed.reshape(-1, ll_mixed.shape[-1])
    flat_below = flat < outlier_log_term
    frac_dominated = flat_below.mean(axis=0)
    std_per_obs = flat_mixed.std(axis=0)

    dominated = frac_dominated > 0.5
    n_dominated = int(dominated.sum())
    print(f"n trials where the outlier term dominates on >50% of draws: "
          f"{n_dominated}", flush=True)

    if n_dominated > 0:
        # subj_idx/trial live as small 1D arrays in /observed_data -- NOT the
        # (chain, draw, obs) broadcast copies inside /log_likelihood, which
        # are the ~800M-element arrays this script specifically avoids
        # loading (see comment above on `node`). The join key is
        # /observed_data's `index` variable, NOT its own `obs_id`: that
        # `obs_id` is just a plain positional 0..n-1 index local to this
        # group, while /log_likelihood's `obs_id` coordinate carries the
        # original (non-contiguous) DataFrame row index -- `index` is the
        # variable that actually matches it (verified: identical value sets).
        with nc4.Dataset(str(nc_path), "r") as f:
            obs_grp = f.groups["observed_data"]
            obs_grp.set_auto_mask(False)
            obs_index = np.asarray(obs_grp.variables["index"][:])
            obs_subj_idx = np.asarray(obs_grp.variables["subj_idx"][:])
            obs_trial = np.asarray(obs_grp.variables["trial"][:])

        obs_lookup = pd.DataFrame(
            {"obs_id": obs_index, "subj_idx": obs_subj_idx, "trial": obs_trial}
        )
        dominated_df = pd.DataFrame({
            "obs_id": obs_id[dominated],
            "frac_dominated": frac_dominated[dominated],
            "std_ll_mixed": std_per_obs[dominated],
        }).merge(obs_lookup, on="obs_id", how="left").sort_values("std_ll_mixed")

        dominated_path = output_dir / f"{model_id}_{dataset}{idx_save}_outlier_dominated.csv"
        dominated_df.to_csv(dominated_path, index=False)
        print(f"Full table saved to {dominated_path}. Preview (most degenerate/"
              f"lowest std first, top 50 of {n_dominated}):", flush=True)
        print(dominated_df.head(50).to_string(index=False), flush=True)

    # LOO is disabled for now: PSIS's Pareto-tail fit crashes with "All tail
    # values are the same" on the outlier-dominated trials printed above.
    # Re-enable once loo_moment_match (or an explicit exclusion strategy) is
    # in place -- see conversation/module docstring for the underlying cause.
    ll_mixed_da = ll_da.copy(data=ll_mixed)
    ll_ds["log_lik"] = ll_mixed_da
    idata_ll = az.convert_to_datatree(ll_ds, group="log_likelihood")
    
    print("  Computing LOO (pointwise) ...", flush=True)
    # reff=1.0 (i.e. assume no autocorrelation) sidesteps az.loo's internal
    # requirement for a `posterior` group to estimate effective sample size --
    # loading the full 40k-draw posterior just for that isn't worth it here.
    # This only affects the reported SE's precision, not the elpd_loo point
    # estimate itself.
    loo_result = az.loo(idata_ll, pointwise=True, reff=1.0, mixture = True)
    print("elpd_loo:", loo_result.elpd, "p_loo:", loo_result.p, flush=True)
    
    k = np.asarray(loo_result.pareto_k.values)
    print("n k>0.7:", (k>0.7).sum(), "max k:", k.max(), "new min ll:", ll_mixed.min(), flush = True)
    
    loo_path = output_dir / f"{model_id}_{dataset}{idx_save}_loo.pkl"
    with open(loo_path, "wb") as f:
        pickle.dump(loo_result, f)

    del ll_ds, ll_da, ll, flat, ll_mixed, flat_mixed

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

    print("  Saving posterior samples ...", flush=True)
    save_posterior_npz(post_ds, model_id, dataset, idx_save, results_pth)

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
        default=0.05,
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

    compute_diagnostics(
        model_id=args.model_id,
        dataset=args.dataset_name,
        results_pth=args.results_pth,
        pupil_col=args.pupil_col,
        p_outlier=args.p_outlier,
        inspect_only=args.inspect_nc,
    )


if __name__ == "__main__":
    main()
