"""Calibrate every experiment in a folder, then combine them."""

import glob
import json
import os

from pyfli import logging

from .calibrator import DetectorSimCalibrator
from .config import CalibrationInputs
from .data_io import MaskShapeError
from .sweep_combination import combine_calibrations, verify_combined

DATA_EXTENSIONS = (".sdt", ".ptu", ".tif", ".tiff", ".npy", ".mat", ".asc")
MASK_EXTENSIONS = (".npy", ".tif", ".tiff", ".mat", ".txt")
MASK_SUFFIXES = ("", "_mask", "-mask", " mask", "_bool_mask", "_final_mask")
AUTOMATIC = "automatic"
REJECTED_MASK_FILE = "rejected_mask.json"


def find_data_files(
    folder: str, extensions=DATA_EXTENSIONS, pattern: str = "*"
) -> list[str]:
    """Raw decay files directly inside ``folder`` (not in sub-folders), sorted by name."""
    files = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(p for p in files if p.lower().endswith(tuple(extensions)))


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def match_masks(
    data_files: list[str],
    mask_folder: str | None,
    extensions=MASK_EXTENSIONS,
    suffixes=MASK_SUFFIXES,
) -> dict[str, str]:
    """
    ``data file -> mask file`` for every data file whose name matches a mask in
    ``mask_folder``: the mask's name (without extension) is the data file's name plus one
    of ``suffixes`` (case-insensitive). With several candidates, the exact name wins.
    Data files without a mask, and masks without a data file, are logged as warnings.
    """
    if mask_folder is None:
        return {}
    if not os.path.isdir(mask_folder):
        raise NotADirectoryError(
            f"mask_folder must be a folder of masks, got {mask_folder}"
        )
    masks = sorted(
        p
        for p in glob.glob(os.path.join(mask_folder, "*"))
        if os.path.isfile(p) and p.lower().endswith(tuple(extensions))
    )
    by_stem = {}
    for m in masks:
        by_stem.setdefault(_stem(m).lower(), []).append(m)
    matched, used = {}, set()
    for d in data_files:
        for suffix in suffixes:
            hits = by_stem.get((_stem(d) + suffix).lower())
            if hits:
                if len(hits) > 1:
                    logging.warning(
                        f"Several masks named {_stem(hits[0])} for {_stem(d)}: using {hits[0]}"
                    )
                matched[d] = hits[0]
                used.add(hits[0])
                break
    missing = [_stem(d) for d in data_files if d not in matched]
    unused = [os.path.basename(m) for m in masks if m not in used]
    if missing:
        logging.warning(
            f"No mask in {mask_folder} for {len(missing)} data file(s), automatic masks are made: {missing}"
        )
    if unused:
        logging.warning(
            f"{len(unused)} mask(s) in {mask_folder} match no data file and are ignored: {unused}"
        )
    return matched


def _check_irf(irf_path: str | None) -> None:
    if irf_path is None:
        return
    if not isinstance(irf_path, (str, os.PathLike)) or os.path.isdir(irf_path):
        raise ValueError(
            "In folder mode irf_path must be ONE IRF file shared by all experiments (same detector), "
            f"not a folder or a list: {irf_path!r}"
        )
    if not os.path.isfile(irf_path):
        raise FileNotFoundError(f"IRF file not found: {irf_path}")


def _reusable(cfg_path: str, mask_path: str | None, irf_path: str | None) -> bool:
    """An existing calibration made with the same mask and IRF."""
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        inp = json.load(f)["inputs"]
    used_mask = inp.get("mask_path") or None
    rejected = os.path.join(os.path.dirname(cfg_path), REJECTED_MASK_FILE)
    if used_mask is None and mask_path is not None and os.path.exists(rejected):
        with open(rejected) as f:
            if json.load(f).get("mask_path") == mask_path:
                used_mask = mask_path
    same = used_mask == mask_path and (inp.get("irf_path") or None) == irf_path
    if not same:
        logging.warning(f"{cfg_path} was made with another mask or IRF: recalibrating")
    return same


def calibrate_folder(
    folder: str,
    detector_type: str,
    model_type: str,
    output_root: str | None = None,
    mask_folder: str | None = None,
    irf_path: str | None = None,
    extensions=DATA_EXTENSIONS,
    pattern: str = "*",
    force: bool = False,
    combine: bool = True,
    weighting: str = "equal",
    verify: bool = True,
    n_verify_pixels: int = 10000,
    **calibration_kwargs,
) -> dict:
    """
    Calibrate each data file in ``folder`` separately, then combine the calibrations.

    ``mask_folder``: folder of masks matched to the data files by name (see the module
    docstring); files without one get an automatic mask. ``irf_path``: one IRF file for all
    experiments, or None to simulate each experiment's IRF. ``calibration_kwargs`` go to
    :class:`~pyfli.simulator.calibration.CalibrationInputs` for every file (e.g. ``bin_radius``,
    ``irf_mode``, ``target_coverage``, budgets); ``mask_path`` and ``reference_path`` are
    per-experiment and not accepted here. A file whose calibration fails is reported in
    ``failed`` and left out of the combination. ``verify`` checks, per experiment, that its
    pixels are covered by its own values and by the whole combined setup
    (:func:`~pyfli.simulator.calibration.sweep_combination.verify_combined`), simulating
    ``n_verify_pixels`` pixels per check.
    """
    for key in ("mask_path", "reference_path"):
        if calibration_kwargs.get(key):
            raise ValueError(
                f"{key} belongs to one experiment; in folder mode use mask_folder (masks matched by name)"
            )
    _check_irf(irf_path)
    files = find_data_files(folder, extensions, pattern)
    if not files:
        raise FileNotFoundError(
            f"No data files with extensions {extensions} in {folder}"
        )
    masks = match_masks(files, mask_folder)
    probe = CalibrationInputs(
        data_path=files[0],
        detector_type=detector_type,
        model_type=model_type,
        **calibration_kwargs,
    )
    if output_root is None:
        output_root = os.path.join(
            os.path.abspath(folder),
            f"sim_calibration_bin{probe.bin_radius}_{probe.detector_type}_{probe.model_type}",
        )
    per_experiment, failed, mask_used = {}, {}, {}
    for i, path in enumerate(files):
        name = _stem(path)
        out_dir = os.path.join(output_root, name)
        cfg_path = os.path.join(out_dir, "calibrated_sim_config.json")
        mask_path = masks.get(path)
        if not force and _reusable(cfg_path, mask_path, irf_path):
            logging.info(
                f"[{i + 1}/{len(files)}] {name}: using the existing calibration"
            )
            with open(cfg_path) as f:
                mask_used[name] = json.load(f)["inputs"].get("mask_path") or AUTOMATIC
            per_experiment[name] = cfg_path
            continue
        logging.info(
            f"[{i + 1}/{len(files)}] {name}: calibrating (mask: {mask_path or AUTOMATIC})"
        )

        def run(mask, name=name, path=path, out_dir=out_dir):
            DetectorSimCalibrator(
                CalibrationInputs(
                    data_path=path,
                    detector_type=detector_type,
                    model_type=model_type,
                    mask_path=mask,
                    irf_path=irf_path,
                    output_dir=out_dir,
                    **calibration_kwargs,
                )
            ).run()

        try:
            try:
                run(mask_path)
                mask_used[name] = mask_path or AUTOMATIC
            except MaskShapeError as exc:
                logging.warning(
                    f"{name}: {exc} ({mask_path}); using an automatic mask instead"
                )
                run(None)
                mask_used[name] = AUTOMATIC
                with open(os.path.join(out_dir, REJECTED_MASK_FILE), "w") as f:
                    json.dump({"mask_path": mask_path, "reason": str(exc)}, f, indent=2)
            per_experiment[name] = cfg_path
        except Exception as exc:
            logging.warning(
                f"{name}: calibration failed ({exc}); left out of the combination"
            )
            failed[name] = repr(exc)
    result = {
        "output_root": output_root,
        "per_experiment": per_experiment,
        "masks": mask_used,
        "irf_path": irf_path,
        "failed": failed,
        "combined": None,
    }
    if combine and per_experiment:
        combined_dir = os.path.join(output_root, "combined")
        combine_calibrations(per_experiment, combined_dir, weighting=weighting)
        result["combined"] = {
            "json": os.path.join(combined_dir, "combined_sim_setup.json"),
            "py": os.path.join(combined_dir, "combined_sim_setup.py"),
        }
        if verify:
            result["coverage"] = verify_combined(
                result["combined"]["json"], n_sim=n_verify_pixels
            )
    return result
