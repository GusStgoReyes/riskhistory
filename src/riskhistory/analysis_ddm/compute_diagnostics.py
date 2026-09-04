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

Consequence for LOO: adding the outlier term back saturates. Any raw
log_lik below about -42 becomes bit-identical to log(p_outlier * W_OUTLIER)
in float64, so trials the DDM finds essentially impossible end up with a
constant corrected log-likelihood across draws. PSIS cannot fit a Pareto
tail to a constant and az.loo raises "All tail values are the same". Those
trials are therefore identified (scan_log_lik) and dropped before az.loo.
Because the dropped set is model-specific, elpd_loo is a sum over a
different number of trials per model, and az.compare -- which requires an
equal observation count across models -- will reject the resulting
ELPDData objects. elpd_i/pareto_k stay labelled by the original obs_id so
compare_models.py can be taught to align models on the common set later.
(Note this arviz version names the pointwise field `elpd_i`, not `loo_i`.)

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


def psis_tail_len(n_draws: int, r_eff: float = 1.0) -> int:
    """Number of draws arviz fits the Pareto tail to, for tail="right".

    Mirrors arviz_stats.base.diagnostics._DiagnosticsBase._get_ps_tails.
    Kept as a local copy rather than imported because that class is
    private; `r_eff` must match the `reff` passed to az.loo below.
    """
    if n_draws * r_eff > 225:
        return int(np.floor(3.0 * (n_draws / r_eff) ** 0.5))
    return int(np.floor(n_draws / 5))


def scan_log_lik(ll_arr, obs_id, outlier_log_term: float, n_tail: int,
                 block: int = 512) -> tuple[pd.DataFrame, dict]:
    """Apply the outlier correction in place and flag PSIS-degenerate trials.

    Walks `ll_arr` (chain, draw, obs_id) in blocks of `block` observations
    so no whole-array temporary is ever materialised -- at 8.3 GB for
    log_lik, a single out-of-place `np.logaddexp` or `.std(axis=0)` is
    enough to OOM the job. `ll_arr[:, :, start:stop]` is a *basic* slice,
    hence a view into the contiguous array, so `np.logaddexp(..., out=view)`
    corrects the array in place with no extra allocation.

    The degeneracy flag reproduces arviz's own check exactly rather than
    approximating it. PSIS fits the Pareto tail to only the `n_tail` most
    extreme draws, and `_ps_tail` raises "All tail values are the same"
    when `abs(max_tail - min_tail) < np.finfo(float).tiny`. After the
    outlier correction, any raw log_lik below about -42 saturates to
    exactly log(p_outlier * W_OUTLIER) -- `exp(ll)` has fallen below the
    float64 resolution of that constant -- so a trial spending even ~1% of
    its draws down there has a bit-constant tail and trips the check.
    Note this is *not* the same set as `frac_dominated > 0.5`: the tail is
    only n_tail of n_draws draws, so a trial can be flagged here while
    being nowhere near outlier-dominated on average, and vice versa.

    Returns per-observation stats (correction already applied to `ll_arr`)
    and a dict of whole-array summaries collected on the *raw* values.
    """
    n_chain, n_draw, n_obs = ll_arr.shape
    n_samples = n_chain * n_draw
    if n_tail >= n_samples:
        raise ValueError(f"n_tail ({n_tail}) must be < n_samples ({n_samples})")

    frac_dominated = np.empty(n_obs)
    tail_ptp = np.empty(n_obs)
    std_ll = np.empty(n_obs)
    n_nan = n_inf = n_below = 0
    raw_min, raw_max = np.inf, -np.inf

    for start in range(0, n_obs, block):
        stop = min(start + block, n_obs)
        width = stop - start
        view = ll_arr[:, :, start:stop]

        # --- raw-value bookkeeping, before the correction ---
        n_nan += int(np.isnan(view).sum())
        n_inf += int(np.isinf(view).sum())
        below = view < outlier_log_term
        n_below += int(below.sum())
        frac_dominated[start:stop] = below.mean(axis=(0, 1))
        del below
        raw_min = min(raw_min, float(np.nanmin(view)))
        raw_max = max(raw_max, float(np.nanmax(view)))

        # --- outlier-mixture correction, in place through the view ---
        np.logaddexp(view, outlier_log_term, out=view)

        # reshape copies (the view is strided over the parent's obs axis),
        # which is the one allocation per block: ~330 MB at block=512.
        blk = view.reshape(-1, width)
        std_ll[start:stop] = blk.std(axis=0)

        # arviz's arithmetic, verbatim: log weights are -log_lik shifted by
        # their own max, and the right tail is the n_tail largest of those
        # (equivalently the n_tail smallest log_lik). np.partition selects
        # the same *set* as arviz's argsort slice, so max - min over it is
        # identical to what _ps_tail compares against tiny.
        lw = -blk
        lw -= lw.max(axis=0)
        part = np.partition(lw, -n_tail, axis=0)[-n_tail:]
        tail_ptp[start:stop] = part.max(axis=0) - part.min(axis=0)
        del blk, lw, part

    stats = pd.DataFrame({
        "obs_id": obs_id,
        "frac_dominated": frac_dominated,
        "tail_ptp": tail_ptp,
        "std_ll_mixed": std_ll,
        "degenerate_tail": tail_ptp < np.finfo(float).tiny,
    })
    summary = {
        "n_nan": n_nan, "n_inf": n_inf, "n_below": n_below,
        "raw_min": raw_min, "raw_max": raw_max,
    }
    return stats, summary


def load_observed_lookup(nc_path: str) -> pd.DataFrame:
    """Map log_likelihood's `obs_id` to subj_idx/trial via /observed_data.

    subj_idx/trial live as small 1D arrays in /observed_data -- NOT the
    (chain, draw, obs) broadcast copies inside /log_likelihood, which are
    the ~1B-element arrays this script specifically avoids loading (see the
    comment on `node` in compute_diagnostics). The join key is
    /observed_data's `index` variable, NOT its own `obs_id`: that `obs_id`
    is just a plain positional 0..n-1 index local to this group, while
    /log_likelihood's `obs_id` coordinate carries the original
    (non-contiguous) DataFrame row index -- `index` is the variable that
    actually matches it (verified: identical value sets).
    """
    with nc4.Dataset(str(nc_path), "r") as f:
        obs_grp = f.groups["observed_data"]
        obs_grp.set_auto_mask(False)
        return pd.DataFrame({
            "obs_id": np.asarray(obs_grp.variables["index"][:]),
            "subj_idx": np.asarray(obs_grp.variables["subj_idx"][:]),
            "trial": np.asarray(obs_grp.variables["trial"][:]),
        })


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

    n_samples = ll_arr.shape[0] * ll_arr.shape[1]
    n_obs_total = ll_arr.shape[2]

    # Correct the log_likelihood for the missing outlier-mixture term (see
    # module docstring): the saved log_lik is log(y * (1 - p_outlier)), so
    # log(exp(log_lik) + p_outlier * W_OUTLIER) recovers
    # log(y * (1 - p_outlier) + w_outlier * p_outlier), the actual mixture
    # density HDDM used at fit time. np.logaddexp is the numerically stable
    # form of this and correctly no-ops when p_outlier == 0 (log(0) = -inf
    # contributes nothing).
    outlier_log_term = np.log(p_outlier * W_OUTLIER) if p_outlier > 0 else -np.inf

    # reff must match the value handed to az.loo below, or the tail length
    # used to flag degenerate trials won't be the one PSIS actually uses.
    reff = 1.0
    n_tail = psis_tail_len(n_samples, reff)

    print(f"Scanning log_likelihood ({ll_arr.shape}, {ll_arr.nbytes / 1e9:.2f} GB), "
          f"applying the outlier correction in place. PSIS tail = {n_tail} "
          f"of {n_samples} draws.", flush=True)
    stats, summary = scan_log_lik(ll_arr, obs_id, outlier_log_term, n_tail)

    print(f"raw log_lik: n nan = {summary['n_nan']}, n inf = {summary['n_inf']}, "
          f"min/max = {summary['raw_min']:.4f}/{summary['raw_max']:.4f}", flush=True)
    print(f"n log_lik values below the mixture floor ({outlier_log_term:.3f}): "
          f"{summary['n_below']} (corrected)", flush=True)

    # Drop the trials whose corrected PSIS tail is bit-constant -- exactly
    # the ones that make az.loo raise "All tail values are the same".
    # frac_dominated > 0.5 is reported alongside for comparison but is NOT
    # the filter: it flags a different (overlapping) set, see scan_log_lik.
    degenerate = stats["degenerate_tail"].to_numpy()
    dominated = stats["frac_dominated"].to_numpy() > 0.5
    print(f"n_obs_total: {n_obs_total} | "
          f"n_dropped_degenerate_tail: {int(degenerate.sum())} | "
          f"n_frac_dominated_gt_0.5: {int(dominated.sum())} | "
          f"overlap: {int((degenerate & dominated).sum())}", flush=True)

    flagged = stats[degenerate | dominated].copy()
    if len(flagged) > 0:
        flagged["dropped"] = flagged["degenerate_tail"]
        flagged = (
            flagged.merge(load_observed_lookup(str(nc_path)), on="obs_id", how="left")
            .sort_values(["dropped", "tail_ptp"], ascending=[False, True])
        )
        flagged_path = output_dir / f"{model_id}_{dataset}{idx_save}_outlier_dominated.csv"
        flagged.to_csv(flagged_path, index=False)
        print(f"Flagged-trial table saved to {flagged_path}. Preview (dropped "
              f"first, most degenerate tail first, top 50 of {len(flagged)}):",
              flush=True)
        print(flagged.head(50).to_string(index=False), flush=True)

    keep = ~degenerate
    # Fancy indexing copies, so this transiently holds both arrays; drop the
    # reference to the full one immediately afterwards so az.loo -- which
    # allocates its own full-shape log_weights and log_weights + log_lik --
    # starts from one copy, not two. Stays float64 on purpose: float32 (eps
    # 1.2e-7) would collapse far more trials into the degenerate set.
    ll_keep = ll_arr[:, :, keep]
    obs_keep = obs_id[keep]
    del ll_arr

    ll_ds = xr.DataArray(
        ll_keep,
        dims=("chain", "draw", "obs_id"),
        coords={"chain": chain, "draw": draw, "obs_id": obs_keep},
        name="log_lik",
    ).to_dataset()
    idata_ll = az.convert_to_datatree(ll_ds, group="log_likelihood")
    
    print(f"  Computing LOO (pointwise) on {len(obs_keep)} retained trials "
          f"({ll_keep.nbytes / 1e9:.2f} GB) ...", flush=True)
    # reff=1.0 (i.e. assume no autocorrelation) sidesteps az.loo's internal
    # requirement for a `posterior` group to estimate effective sample size --
    # loading the full 80k-draw posterior just for that isn't worth it here.
    # This only affects the reported SE's precision, not the elpd_loo point
    # estimate itself. mixture stays False: that flag selects Mix-IS-LOO
    # (Silva & Zanella 2023), which assumes the draws came from a mixture of
    # leave-one-out *posteriors* -- a property of the sampling scheme,
    # unrelated to HDDM's outlier-mixture likelihood.
    try:
        loo_result = az.loo(idata_ll, pointwise=True, reff=reff, mixture=False)
    except ValueError as err:
        if "All tail values are the same" not in str(err):
            raise
        # The drop criterion replicates arviz's check exactly, so reaching
        # here means the replication has drifted from the installed arviz.
        kept_ptp = stats.loc[keep, "tail_ptp"].to_numpy()
        kept_obs = stats.loc[keep, "obs_id"].to_numpy()
        raise RuntimeError(
            f"PSIS still reports a constant tail after dropping "
            f"{int(degenerate.sum())} trials ({len(obs_keep)} retained). "
            f"Smallest retained tail_ptp = {kept_ptp.min():.3e} at obs_id "
            f"{kept_obs[kept_ptp.argmin()]}. Check psis_tail_len/scan_log_lik "
            f"against arviz's _ps_tail."
        ) from err

    print("elpd_loo:", loo_result.elpd, "p_loo:", loo_result.p,
          "n_data_points:", loo_result.n_data_points, flush=True)
    
    k = np.asarray(loo_result.pareto_k.values)
    print("n k>0.7:", (k > 0.7).sum(), "max k:", k.max(),
          "min corrected ll:", ll_keep.min(), flush=True)
    
    # az.loo attaches the full-shape PSIS log_weights to the returned
    # ELPDData (arviz_stats/loo/helper_loo.py, ELPDData(..., log_weights=)),
    # which makes the pickle as large as log_lik itself -- 8.3 GB per model,
    # i.e. >100 GB across ALL_MODELS, and compare_models.py loads a whole
    # group at once. Nothing downstream reads it (az.compare never touches
    # log_weights; compare_models.py uses elpd/se/p/elpd_i/pareto_k), and it
    # is recomputable from the .nc, so drop it before writing.
    loo_result.log_weights = None

    loo_path = output_dir / f"{model_id}_{dataset}{idx_save}_loo.pkl"
    with open(loo_path, "wb") as f:
        pickle.dump(loo_result, f)

    del ll_ds, idata_ll, ll_keep, loo_result

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
