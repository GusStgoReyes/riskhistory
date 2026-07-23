"""Extract posterior parameter samples from a fitted HDDM run into a compact numpy file.

Opens only the 'posterior' netCDF4 group of the InferenceData file (never touching
the much larger 'log_likelihood'/'observed_data' groups used for LOO in
compute_diagnostics.py), and saves one array per parameter into a compressed .npz.
The result is small enough to download and plot/analyze locally with plain numpy --
no hddm, arviz, or xarray installation required to read it back.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def extract_posterior(
    model_id: str,
    dataset: str,
    results_pth: str,
    output_pth: str = None,
    dtype=np.float32,
    pupil_col: str = "pupil",
) -> Path:
    """Extract posterior samples for one fitted model into a compressed .npz.

    Parameters
    ----------
    model_id : str
        Model identifier (matches the .nc filename in results_pth/hddm/).
    dataset : str
        Dataset name ('equalIndifference', 'equalRange', 'sheng2020').
    results_pth : str
        Path to the results directory (contains the 'hddm' subdirectory).
    output_pth : str, optional
        Directory to save the .npz. Defaults to results_pth/hddm_posteriors.
    dtype : optional
        Dtype to cast samples to before saving (default float32, halves size
        relative to the float64 samples HDDM stores).
    pupil_col : str, optional
        Which pupil column the fit used as its "pupil" regressor -- must
        match how the model was fit (model_hddm.py), since it determines the
        filename suffix.

    Returns
    -------
    Path to the saved .npz file.
    """
    # Must match model_hddm.py's save_name / utils_stats.py's idx_save suffix.
    idx_save = "" if pupil_col == "pupil" else f"_{pupil_col}"

    hddm_dir = Path(results_pth) / "hddm"
    nc_path = hddm_dir / f"{model_id}_{dataset}{idx_save}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    # Group-scoped open: only 'posterior' is read off disk, so the multi-GB
    # log_likelihood/observed_data groups in the same file are never loaded.
    posterior = xr.open_dataset(nc_path, group="posterior")

    arrays = {var: da.values.astype(dtype) for var, da in posterior.data_vars.items()}
    arrays["_chain"] = posterior["chain"].values
    arrays["_draw"] = posterior["draw"].values

    output_dir = Path(output_pth) if output_pth else Path(results_pth) / "hddm_posteriors"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_id}_{dataset}{idx_save}_posterior.npz"

    np.savez_compressed(out_path, **arrays)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {len(arrays) - 2} parameters to {out_path} ({size_mb:.1f} MB)", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract posterior parameter samples from a fitted HDDM run into a .npz"
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="equalIndifference",
        choices=["equalIndifference", "sheng2020", "equalRange"],
    )
    parser.add_argument(
        "--results_pth", type=str, required=True, help="Path to results directory"
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        default=None,
        help="Directory to save the .npz (default: {results_pth}/hddm_posteriors)",
    )
    parser.add_argument(
        "--pupil_col",
        type=str,
        default="pupil",
        choices=["pupil", "pupil_residual"],
        help="Which pupil column the fit used as its 'pupil' regressor "
             "(must match how the model was fit, for filename lookup)",
    )
    args = parser.parse_args()

    extract_posterior(
        model_id=args.model_id,
        dataset=args.dataset_name,
        results_pth=args.results_pth,
        output_pth=args.output_pth,
        pupil_col=args.pupil_col,
    )


if __name__ == "__main__":
    main()
