"""Subset extracted posterior .npz files down to a few parameters of interest.
"""
import argparse
from pathlib import Path

import numpy as np

# Parameters to keep per model, as they appear in the posterior (HDDM's patsy
# names, so interactions use ':'). Each stem also brings along its '_std' and
# all per-subject 'stem_subj.<id>' arrays.
MODEL_PARAMS = {
    "HDDM_Base_Lag1-drift": ["v_choice_prev1"],
    "HDDM_Base_Lag1-stp": ["z_choice_prev1"],
    "HDDM_Base_Lag1-stim": ["v_gain_centered", "v_gain_centered:choice_prev1", "v_loss_centered:choice_prev1", "v_loss_centered"],
    "HDDM_Base_Lag1-full" : ["v_choice_prev1", "z_choice_prev1",
                             "v_gain_centered", "v_gain_centered:choice_prev1", 
                             "v_loss_centered:choice_prev1", "v_loss_centered"],
    "HDDM_Base_Lag1-stp_Pupil_stp": ["z_choice_prev1:pupil", "z_pupil"],
    "HDDM_Base_Lag1-drift_Pupil_drift": ["v_choice_prev1:pupil", "v_pupil"],
    "HDDM_Base_Pupil_valuation": [
        "v_gain_centered:pupil",
        "v_loss_centered:pupil",
        "v_pupil",
    ],
    "HDDM_Base_Pupil_boundary": ["a_pupil"],
    "HDDM_Base_Pupil_ndt": ["t_pupil"],
}

DATASETS = ("equalIndifference", "equalRange", "sheng2020")


def subset_posterior(
    npz_path: str | Path,
    output_pth: str | Path,
    params,
    keep_group_level: bool = True,
) -> Path:
    """Rewrite one posterior .npz keeping only the arrays for `params`.

    Parameters
    ----------
    npz_path : str or Path
        Path to a posterior .npz written by extract_posteriors.py.
    output_pth : str or Path
        Directory to write the subset .npz into (created if needed). The
        filename is reused, so this must differ from the input directory.
    params : sequence of str
        Parameter stems to keep, e.g. ['v_choice_prev1:pupil', 'v_pupil'].
        For each stem, every `<stem>_subj.<id>` array is kept.
    keep_group_level : bool, optional
        Also keep the group-level `<stem>` and `<stem>_std` arrays. Default True.

    Returns
    -------
    Path to the saved .npz file.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Posterior file not found: {npz_path}")

    output_dir = Path(output_pth)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / npz_path.name
    if out_path.resolve() == npz_path.resolve():
        raise ValueError(f"Output would overwrite the input file: {out_path}")

    subj_prefixes = tuple(f"{p}_subj." for p in params)
    group_keys = set()
    if keep_group_level:
        for p in params:
            group_keys.update({p, f"{p}_std"})

    with np.load(npz_path, allow_pickle=True) as npz:
        available = set(npz.files)
        missing = group_keys - available
        if missing:
            raise KeyError(
                f"{npz_path.name} is missing expected keys: {sorted(missing)}. "
                f"Check the parameter names in MODEL_PARAMS."
            )
        subj_keys = [k for k in npz.files if k.startswith(subj_prefixes)]
        if not subj_keys:
            raise KeyError(
                f"No per-subject keys in {npz_path.name} match {subj_prefixes}"
            )
        keys = sorted(group_keys) + subj_keys + [
            k for k in ("_chain", "_draw") if k in available
        ]
        # Only these keys are decompressed off disk.
        arrays = {k: npz[k] for k in keys}

    np.savez_compressed(out_path, **arrays)
    size_mb = out_path.stat().st_size / 1e6
    print(
        f"{npz_path.name}: kept {len(arrays)} keys "
        f"({len(subj_keys)} subject-level) -> {out_path} ({size_mb:.1f} MB)",
        flush=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Subset extracted HDDM posterior .npz files to a few parameters"
    )
    parser.add_argument(
        "--model_ids",
        type=str,
        nargs="+",
        default=list(MODEL_PARAMS),
        choices=list(MODEL_PARAMS),
        help="Models to process (default: all in MODEL_PARAMS)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=list(DATASETS), choices=DATASETS
    )
    parser.add_argument(
        "--posteriors_pth",
        type=str,
        default="results/analysis_ddm_final/hddm_posteriors",
        help="Directory holding the full posterior .npz files",
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        default=None,
        help="Directory to write subsets (default: {posteriors_pth}/smaller_posteriors)",
    )
    args = parser.parse_args()

    pth = Path(args.posteriors_pth)
    output_pth = Path(args.output_pth) if args.output_pth else pth / "smaller_posteriors"

    for model_id in args.model_ids:
        for dataset in args.datasets:
            subset_posterior(
                npz_path=pth / f"{model_id}_{dataset}_posterior.npz",
                output_pth=output_pth,
                params=MODEL_PARAMS[model_id],
            )


if __name__ == "__main__":
    main()
