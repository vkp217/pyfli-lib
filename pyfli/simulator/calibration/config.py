"""Inputs and constants for calibrating the pyfli simulator against an experimental cube."""

from dataclasses import dataclass, field

DETECTOR_TYPES = ("continuous", "discrete")
MODEL_TYPES = ("bi_exponential", "mono_exponential")
REFERENCE_SOURCES = ("auto", "alligator", "spcimage", "pyfli")

# pyfli.simulator constants that the calibration has to respect.
# Only the "separate" family (ContinuousSimulator / PhotonCountSimulator) supports a Beta
# prior on tau2; the "combined" family draws tau2 from a truncated normal.
SIM_FAMILY = "separate"
SIM_MU_PER_CYCLE = 0.01  # BaseDiscreteSimulator: expected photons per laser cycle
SIM_MIN_CYCLES = 1000  # BaseFLIEngine._normalize_cycles_range lower bound
SIM_FRACTION_SCALE = 0.998  # FLIModelSimulator: E / A1 = Beta * 0.998 + 0.001
SIM_FRACTION_OFFSET = 0.001
SIM_QE_RANGE = {
    "continuous": (0.15, 0.35),
    "discrete": (0.70, 0.90),
}  # ParameterSampler.sample_qe


def normalize_model_type(model_type: str) -> str:
    key = model_type.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bi": "bi_exponential",
        "bi_exponential": "bi_exponential",
        "biexponential": "bi_exponential",
        "mono": "mono_exponential",
        "mono_exponential": "mono_exponential",
        "monoexponential": "mono_exponential",
    }
    if key not in aliases:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    return aliases[key]


def normalize_detector_type(detector_type: str) -> str:
    key = detector_type.lower().strip()
    key = {"continous": "continuous", "tcspc": "discrete", "iccd": "continuous"}.get(
        key, key
    )
    if key not in DETECTOR_TYPES:
        raise ValueError(
            f"detector_type must be one of {DETECTOR_TYPES}, got {detector_type!r}"
        )
    return key


@dataclass
class PixelSelectionCriteria:
    """
    Thresholds that decide whether a pixel carries sample fluorescence or only
    detector/background noise. See :mod:`pixel_selection` for the tests.
    """

    min_signal_photons: float = 50.0  # background-subtracted counts per pixel
    snr_min: float = 5.0  # signal / sqrt(total): Poisson significance above background
    flatness_sigma: float = (
        5.0  # reject decays statistically consistent with a flat (dark) trace
    )
    peak_window_bins: int | None = (
        None  # |peak - global peak| tolerance; None -> n_bins // 8
    )
    use_otsu: str = "auto"  # "auto" | "on" | "off"; auto applies Otsu only to clearly bimodal images
    otsu_min_separability: float = 0.6
    hot_pixel_factor: float = 10.0  # intensity > factor * 3x3 median -> hot pixel
    min_component_size: int = 9  # drop isolated specks smaller than this (pixels)
    refine_user_mask: bool = (
        False  # True: intersect a provided mask with the automatic criteria
    )


@dataclass
class CalibrationInputs:
    """
    Everything the calibration needs.

    Parameters
    ----------
    data_path : str
        Raw 3D decay data (.sdt, .ptu, .tif, .npy, ...), loaded via
        ``pyfli.io.DataOperations``.
    detector_type : str
        "continuous" (gated/ICCD-like) or "discrete" (TCSPC photon counting).
    model_type : str
        "bi_exponential" or "mono_exponential".
    irf_path : str | None
        Experimental IRF. When omitted the IRF is simulated with
        ``pyfli.simulator.IRFGenerator.gaussianExpIRF`` and its parameters are optimised.
    mask_path : str | None
        Boolean mask of sample pixels. When omitted, one is derived automatically.
    reference_path : str | None
        Optional parameter maps (AlliGator folder, SPCImage folder, or a pyfli
        session .npy file / folder) used to initialise and regularise the priors.
    target_coverage : float
        Required fraction of experimental pixels inside the simulated distribution for
        every coverage statistic (raw phasors h=1/2, photons, rising edge, background); see
        :mod:`coverage`. After the best-fit calibration the config is widened until this is
        reached, so that the experiment is a subset of what the simulator produces. 0 keeps
        the best-fit config.
    n_coverage_pixels : int
        Simulated pixels per coverage check.
    max_coverage_iter : int
        Maximum number of widening steps.
    """

    data_path: str
    detector_type: str
    model_type: str
    irf_path: str | None = None
    irf_mode: str = "auto"  # "experimental" | "simulated" | "auto" (given IRF vs fitted EMG: better pooled fit wins)
    bg_path: str | None = None
    mask_path: str | None = None
    reference_path: str | None = None
    reference_source: str = "auto"
    reference_alpha: str = "auto"  # "intensity" | "amplitude" | "auto" (per source)
    laser_freq_mhz: float = 80.0
    bin_radius: int = (
        0  # spatial binning (pyfli FLIBinner); 3 matches the *bin3* sessions
    )
    bit_depth: int | None = None  # continuous only; None -> inferred from the data
    jitter: bool | str = (
        "auto"  # simulator's +/-2 bin jitter; "auto": off when the measured edge spread is too narrow
    )
    mono_fraction: float | None = None  # None -> 0 for bi, 1 for mono
    prior_inflation: float = (
        1.0  # >1 widens the fitted Beta priors (variance multiplier)
    )
    optimize_irf: bool = True
    optimize_noise: bool = True
    optimize_priors: bool = True
    reference_weight: float = (
        0.05  # penalty keeping optimised prior means near the reference maps
    )
    n_exp_pixels: int = 20000  # experimental pixels used for summary statistics
    n_sim_pixels: int = 2000  # simulated pixels per objective evaluation
    n_fit_pixels: int = 1500  # pixels for the quick per-pixel fits (no reference maps)
    n_validation_pixels: int = 6000
    n_bank_pixels: int = 20000  # importance-weighted bank for the prior fit (stage 2)
    bank_rounds: int = 3  # alternating prior (bank) / IRF-noise rounds
    max_evals_stage1: int = (
        120  # IRF + detector noise, real simulations (later rounds use half)
    )
    max_evals_stage2: int = (
        1500  # Beta priors on the re-weighted bank (~ms each), per round
    )
    target_coverage: float = 0.99
    n_coverage_pixels: int = 20000
    max_coverage_iter: int = 10
    seed: int = 0
    output_dir: str | None = None
    selection: PixelSelectionCriteria = field(default_factory=PixelSelectionCriteria)

    def __post_init__(self) -> None:
        self.detector_type = normalize_detector_type(self.detector_type)
        self.model_type = normalize_model_type(self.model_type)
        if self.reference_source not in REFERENCE_SOURCES:
            raise ValueError(f"reference_source must be one of {REFERENCE_SOURCES}")
        if self.mono_fraction is None:
            self.mono_fraction = 1.0 if self.model_type == "mono_exponential" else 0.0

    @property
    def laser_period_ns(self) -> float:
        return 1000.0 / self.laser_freq_mhz

    @property
    def is_bi(self) -> bool:
        return self.model_type == "bi_exponential"
