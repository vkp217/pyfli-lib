"""Command-line calibration of the pyfli simulator against an experimental decay cube."""

import argparse
import os

from .calibrator import DetectorSimCalibrator
from .config import CalibrationInputs, PixelSelectionCriteria
from .folder_calibration import calibrate_folder


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data", required=True, help="raw 3D decay data (.sdt/.ptu/.tif/.npy/...)"
    )
    p.add_argument("--detector", required=True, choices=["continuous", "discrete"])
    p.add_argument(
        "--model", required=True, choices=["bi_exponential", "mono_exponential"]
    )
    p.add_argument(
        "--irf",
        default=None,
        help="experimental IRF (one file, also in folder mode); omit to simulate it",
    )
    p.add_argument(
        "--irf-mode", default="auto", choices=["auto", "experimental", "simulated"]
    )
    p.add_argument("--bg", default=None, help="background file/folder")
    p.add_argument(
        "--mask",
        default=None,
        help="boolean sample mask (folder mode: folder of masks named like the data files); omit for automatic",
    )
    p.add_argument(
        "--reference",
        default=None,
        help="AlliGator / SPCImage folder or pyfli session .npy/folder",
    )
    p.add_argument(
        "--reference-source",
        default="auto",
        choices=["auto", "alligator", "spcimage", "pyfli"],
    )
    p.add_argument(
        "--reference-alpha", default="auto", choices=["auto", "intensity", "amplitude"]
    )
    p.add_argument("--laser-mhz", type=float, default=80.0)
    p.add_argument("--bin-radius", type=int, default=0)
    p.add_argument(
        "--bit-depth", type=int, default=None, help="continuous detectors only"
    )
    p.add_argument("--jitter", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--mono-fraction", type=float, default=None)
    p.add_argument("--prior-inflation", type=float, default=1.0)
    p.add_argument("--reference-weight", type=float, default=0.05)
    p.add_argument("--no-optimize-irf", action="store_true")
    p.add_argument("--no-optimize-noise", action="store_true")
    p.add_argument("--no-optimize-priors", action="store_true")
    p.add_argument("--n-sim", type=int, default=2000)
    p.add_argument("--n-exp", type=int, default=20000)
    p.add_argument("--n-fit", type=int, default=1500)
    p.add_argument(
        "--n-bank",
        type=int,
        default=20000,
        help="simulated pixels in the prior-fit bank",
    )
    p.add_argument(
        "--rounds", type=int, default=3, help="alternating prior / IRF-noise rounds"
    )
    p.add_argument(
        "--evals1",
        type=int,
        default=120,
        help="IRF / noise evaluations (first round; later rounds half)",
    )
    p.add_argument(
        "--evals2",
        type=int,
        default=1500,
        help="prior evaluations per round (cheap, bank re-weighting)",
    )
    p.add_argument(
        "--target-coverage",
        type=float,
        default=0.99,
        help="fraction of experimental pixels that must lie inside the simulated distribution (0: best fit only)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--weighting",
        default="equal",
        choices=["equal", "pixels"],
        help="folder mode: weight of each experiment in the combined setup",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="folder mode: recalibrate experiments that already have a config",
    )
    # pixel selection
    p.add_argument("--min-signal", type=float, default=50.0)
    p.add_argument("--snr-min", type=float, default=5.0)
    p.add_argument("--flatness-sigma", type=float, default=5.0)
    p.add_argument("--otsu", default="auto", choices=["auto", "on", "off"])
    p.add_argument(
        "--refine-mask",
        action="store_true",
        help="intersect --mask with the automatic criteria",
    )
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    shared = dict(
        irf_mode=a.irf_mode,
        bg_path=a.bg,
        reference_source=a.reference_source,
        reference_alpha=a.reference_alpha,
        laser_freq_mhz=a.laser_mhz,
        bin_radius=a.bin_radius,
        bit_depth=a.bit_depth,
        jitter={"auto": "auto", "on": True, "off": False}[a.jitter],
        mono_fraction=a.mono_fraction,
        prior_inflation=a.prior_inflation,
        reference_weight=a.reference_weight,
        optimize_irf=not a.no_optimize_irf,
        optimize_noise=not a.no_optimize_noise,
        optimize_priors=not a.no_optimize_priors,
        n_sim_pixels=a.n_sim,
        n_exp_pixels=a.n_exp,
        n_fit_pixels=a.n_fit,
        n_bank_pixels=a.n_bank,
        bank_rounds=a.rounds,
        max_evals_stage1=a.evals1,
        max_evals_stage2=a.evals2,
        target_coverage=a.target_coverage,
        seed=a.seed,
        selection=PixelSelectionCriteria(
            min_signal_photons=a.min_signal,
            snr_min=a.snr_min,
            flatness_sigma=a.flatness_sigma,
            use_otsu=a.otsu,
            refine_user_mask=a.refine_mask,
        ),
    )
    if os.path.isdir(a.data):
        if a.reference:
            raise SystemExit(
                "--reference belongs to one experiment; it cannot be used with a data folder"
            )
        if a.mask and not os.path.isdir(a.mask):
            raise SystemExit(
                "With a data folder, --mask must be a folder of masks named like the data files"
            )
        if a.irf and not os.path.isfile(a.irf):
            raise SystemExit(
                "With a data folder, --irf must be one IRF file shared by all experiments"
            )
        out = calibrate_folder(
            a.data,
            a.detector,
            a.model,
            output_root=a.out,
            mask_folder=a.mask,
            irf_path=a.irf,
            force=a.force,
            weighting=a.weighting,
            **shared,
        )
        print(
            f"\n{len(out['per_experiment'])} experiments calibrated (one config each):"
        )
        for name, path in out["per_experiment"].items():
            print(f"  {name}: {path}  (mask: {out['masks'][name]})")
        for name, err in out["failed"].items():
            print(f"  FAILED {name}: {err}")
        if out["combined"]:
            print(f"combined setup: {out['combined']['py']}")
        return
    inputs = CalibrationInputs(
        data_path=a.data,
        detector_type=a.detector,
        model_type=a.model,
        mask_path=a.mask,
        irf_path=a.irf,
        reference_path=a.reference,
        output_dir=a.out,
        **shared,
    )
    result = DetectorSimCalibrator(inputs).run()
    print("\nCalibrated pyfli simulator config:")
    for k, v in result.config.items():
        print(f"  {k}: {v}")
    print(f"SimGenerator kwargs: {result.sim_generator_kwargs}")
    v = result.validation
    print(
        f"distance initial -> best fit -> covering: {v['distance_initial']:.3f} -> {v['distance_best_fit']:.3f} -> {v['distance_final']:.3f}"
    )
    print(f"coverage of the experiment: {v['coverage_final']}")
    print(f"FLIValidator: {v['fli_validator']}")
    print(f"outputs: {result.output_dir}")


if __name__ == "__main__":
    main()
