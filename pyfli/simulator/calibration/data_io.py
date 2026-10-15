"""Load the experimental decay cube, IRF, mask and AlliGator / SPCImage / pyfli maps."""

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from pyfli import logging
from pyfli.io import AlliGprocessedImport, BHprocessedImport, DataOperations
from pyfli.solver.binned_fitter import FLIBinner

# Preferred pyfli session outputs, most trusted first (all are pixel-wise fits)
PYFLI_REFERENCE_PATTERNS = (
    "CPU_MLE_poisson_*.npy",
    "F-BI*_Output.npy",
    "Bi Direct_Output_*.npy",
    "CPU_NLSF_*.npy",
    "*Direct_Output_*.npy",
)


@dataclass
class ExperimentalData:
    decay: np.ndarray  # (H, W, T) float32
    irf: np.ndarray | None  # (T,) or (H, W, T), or None
    mask: np.ndarray | None  # (H, W) bool provided by the user, or None
    source_path: str
    bin_radius: int = 0

    @property
    def n_bins(self) -> int:
        return self.decay.shape[-1]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.decay.shape[:2]


@dataclass
class ReferenceMaps:
    source: str  # "alligator" | "spcimage" | "pyfli"
    path: str
    maps: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # tau1, tau2, alpha1, efficiency, tau, photon_count

    def aligned_with(self, spatial_shape: tuple[int, int]) -> bool:
        return all(v.shape == tuple(spatial_shape) for v in self.maps.values())

    def values(self, key: str, mask: np.ndarray | None = None) -> np.ndarray:
        """Finite values of one map, restricted to ``mask`` when the shapes agree."""
        if key not in self.maps:
            return np.array([])
        m = self.maps[key]
        if mask is not None and m.shape == mask.shape:
            m = m[mask]
        m = np.asarray(m, dtype=float).ravel()
        return m[np.isfinite(m)]


# ----------------------------------------------------------------------------- raw data
def _as_cube(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:  # (N, T) list of decays
        return arr[None, :, :]
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 4:  # (C, H, W, T) or (H, W, T, C): keep the first channel
        logging.warning(f"4D data {arr.shape}: using the first channel")
        return arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
    raise ValueError(f"Unsupported decay data shape {arr.shape}")


class MaskShapeError(ValueError):
    """The mask's spatial shape differs from the decay cube's."""


def _load_mask(mask_path: str, spatial_shape: tuple[int, int]) -> np.ndarray:
    if mask_path.lower().endswith(".npy"):
        mask = np.load(mask_path, allow_pickle=True)
        mask = (
            mask.astype(bool)
            if mask.dtype != object
            else np.asarray(mask.item(), dtype=bool)
        )
    else:
        mask = DataOperations(mask_path=mask_path).load_mask()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.any(-1)
    if mask.shape != tuple(spatial_shape):
        raise MaskShapeError(
            f"Mask shape {mask.shape} does not match the data {spatial_shape}"
        )
    return mask.astype(bool)


def load_experimental_data(
    data_path: str,
    irf_path: str | None = None,
    bg_path: str | None = None,
    mask_path: str | None = None,
    bin_radius: int = 0,
) -> ExperimentalData:
    """
    Load the 3D decay cube (and optional IRF / mask) with ``pyfli.io.DataOperations``
    and optionally apply pyfli's spatial binning (``FLIBinner``). Binning a provided
    mask is not needed: it stays in the original pixel grid.
    """
    ops = DataOperations(data_path=data_path, irf_path=irf_path, bg_path=bg_path)
    decay = ops.load_data(sub_bg=bg_path is not None)
    if decay is None:
        raise FileNotFoundError(f"Could not load decay data from {data_path}")
    decay = _as_cube(decay)
    decay = np.nan_to_num(np.clip(decay, 0, None))

    irf = None
    if irf_path:
        irf = ops.load_irf(sub_bg=False)
        if irf is None:
            raise FileNotFoundError(f"Could not load IRF from {irf_path}")
        irf = np.asarray(irf, dtype=np.float32)
        if irf.ndim == 2 and irf.shape[0] == 1:
            irf = irf[0]
        if irf.shape[-1] != decay.shape[-1]:
            raise ValueError(
                f"IRF has {irf.shape[-1]} time bins, data has {decay.shape[-1]}"
            )
        if irf.ndim == 3 and irf.shape[:2] != decay.shape[:2]:
            logging.warning(
                "IRF cube spatial shape differs from data: using its mean trace"
            )
            irf = irf.reshape(-1, irf.shape[-1]).mean(0)

    if bin_radius > 0:
        irf_cube = (
            irf
            if (irf is not None and irf.ndim == 3)
            else np.zeros((*decay.shape[:2], 1), np.float32)
        )
        decay, binned_irf = FLIBinner(bin_radius).apply_binning(decay, irf_cube)
        if irf is not None and irf.ndim == 3:
            irf = binned_irf

    mask = _load_mask(mask_path, decay.shape[:2]) if mask_path else None
    logging.info(
        f"Loaded decay {decay.shape} (bin radius {bin_radius}); IRF: "
        f"{'none' if irf is None else irf.shape}; mask: {'none' if mask is None else int(mask.sum())} px"
    )
    return ExperimentalData(
        decay=decay, irf=irf, mask=mask, source_path=data_path, bin_radius=bin_radius
    )


# ------------------------------------------------------------------------ reference maps
def _detect_source(path: str) -> str:
    if os.path.isfile(path):
        return "pyfli"
    if glob.glob(os.path.join(path, "*Map.txt")):
        return "alligator"
    if glob.glob(os.path.join(path, "*.asc")):
        return "spcimage"
    if any(glob.glob(os.path.join(path, p)) for p in PYFLI_REFERENCE_PATTERNS):
        return "pyfli"
    raise FileNotFoundError(f"No AlliGator, SPCImage or pyfli maps found in {path}")


def _to_ns(tau: np.ndarray) -> np.ndarray:
    """SPCImage exports lifetimes in ps; anything with a median above 50 is treated as ps."""
    med = np.nanmedian(tau[np.isfinite(tau) & (tau > 0)]) if np.any(tau > 0) else 0
    return tau / 1000.0 if med > 50 else tau


def _amplitude_to_intensity(
    a1: np.ndarray, tau1: np.ndarray, tau2: np.ndarray
) -> np.ndarray:
    num = a1 * tau1
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / (num + (1 - a1) * tau2)


def _canonical(raw: dict[str, np.ndarray], alpha_kind: str) -> dict[str, np.ndarray]:
    """Map loader-specific keys to tau1/tau2/alpha1/efficiency/tau/photon_count."""

    def get(*keys):
        return next((np.asarray(raw[k], float) for k in keys if k in raw), None)

    out = {}
    tau1, tau2 = get("tau1_map", "tau1"), get("tau2_map", "tau2")
    tau = get("tau_map", "tau")
    pc = get("photon_count_map", "photon_count")
    if pc is not None:
        out["photon_count"] = pc

    if tau1 is not None and tau2 is not None:
        tau1, tau2 = _to_ns(tau1), _to_ns(tau2)
        a1 = None
        if alpha_kind == "intensity":
            a1 = get("alpha1_map", "alpha1", "f1i_map")
        if a1 is None:  # amplitude fractions -> intensity fractions
            amp = get("f1a_map", "alpha1_map", "alpha1")
            if amp is None and "A1_perc_map" in raw:
                amp = np.asarray(raw["A1_perc_map"], float) / 100.0
            if amp is None and "A1_map" in raw and "A2_map" in raw:
                a, b = (
                    np.asarray(raw["A1_map"], float),
                    np.asarray(raw["A2_map"], float),
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    amp = a / (a + b)
            if amp is not None:
                a1 = _amplitude_to_intensity(amp, tau1, tau2)
        swap = tau1 > tau2  # enforce tau1 = short component
        tau1, tau2 = np.where(swap, tau2, tau1), np.where(swap, tau1, tau2)
        valid = np.isfinite(tau1) & np.isfinite(tau2) & (tau1 > 0) & (tau2 > 0)
        out["tau1"] = np.where(valid, tau1, np.nan)
        out["tau2"] = np.where(valid, tau2, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["efficiency"] = np.where(valid, 1 - tau1 / tau2, np.nan)
        if a1 is not None:
            a1 = np.where(swap, 1 - a1, a1)
            out["alpha1"] = np.where(valid & (a1 >= 0) & (a1 <= 1), a1, np.nan)
        with np.errstate(invalid="ignore"):
            out["tau"] = out["tau1"] * out.get("alpha1", 0.5) + out["tau2"] * (
                1 - out.get("alpha1", 0.5)
            )
    elif tau is not None:
        tau = _to_ns(tau)
        out["tau"] = np.where(np.isfinite(tau) & (tau > 0), tau, np.nan)
    return out


def _load_pyfli_npy(path: str) -> dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    obj = obj.item() if obj.dtype == object else obj
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a pyfli map dictionary")
    if "results" in obj:
        obj = obj["results"].get("maps", obj["results"])
    return {k: v for k, v in obj.items() if isinstance(v, np.ndarray) and v.ndim == 2}


def load_reference_maps(
    path: str, source: str = "auto", alpha_kind: str = "auto"
) -> ReferenceMaps:
    """
    Load optional parameter maps from AlliGator (``*Map.txt`` folder), SPCImage
    (``*.asc`` folder) or pyfli (a session ``.npy`` file, or a session folder in which
    case the first match of :data:`PYFLI_REFERENCE_PATTERNS` is used).

    ``alpha_kind`` says whether the source's fraction is an intensity fraction (pyfli,
    AlliGator ``f1_i``) or an amplitude fraction (SPCImage ``a1[%]``, AlliGator ``f1_a``).
    """
    source = _detect_source(path) if source == "auto" else source
    if source == "alligator":
        raw = AlliGprocessedImport(path)._read_text_files() or {"results": {"maps": {}}}
        raw = raw["results"]["maps"]
        kind = "intensity" if alpha_kind == "auto" else alpha_kind
    elif source == "spcimage":
        raw = BHprocessedImport(path)._read_ascfiles() or {"results": {"maps": {}}}
        raw = raw["results"]["maps"]
        kind = "amplitude" if alpha_kind == "auto" else alpha_kind
    else:
        file_path = path
        if os.path.isdir(path):
            hits = [
                h
                for p in PYFLI_REFERENCE_PATTERNS
                for h in sorted(glob.glob(os.path.join(path, p)))
            ]
            file_path = hits[0]
        raw = _load_pyfli_npy(file_path)
        path = file_path
        kind = "intensity" if alpha_kind == "auto" else alpha_kind

    maps = _canonical(raw, kind)
    if not maps:
        raise ValueError(f"No lifetime maps could be read from {path}")
    logging.info(
        f"Reference maps ({source}, alpha1 as {kind} fraction) from {path}: {sorted(maps)}"
    )
    return ReferenceMaps(source=source, path=path, maps=maps)
