from typing import Any

import numpy as np

from .combined.main_factory import MacroSimulator, TCSPCSimulator
from .irf_sim.irf_offset_gen import OffsetGen
from .separate.main_factory_gen import ContinuousSimulator, PhotonCountSimulator

SIMULATOR_TYPES = {
    "combined": {
        "continuous": MacroSimulator,
        "discrete": TCSPCSimulator,
    },
    "separate": {
        "continuous": ContinuousSimulator,
        "discrete": PhotonCountSimulator,
    },
}


def concat_sim_data(*datasets):
    """
    Concatenate multiple sim.sample() output dicts along the batch axis (axis=0).
    Assumes all datasets share identical keys and per-sample shapes.
    """
    keys = datasets[0].keys()
    # sanity check: all datasets must have the same keys
    for i, d in enumerate(datasets[1:], start=1):
        if set(d.keys()) != set(keys):
            missing = set(keys) - set(d.keys())
            extra = set(d.keys()) - set(keys)
            raise ValueError(
                f"Key mismatch in dataset {i}: missing={missing}, extra={extra}"
            )

    combined = {}
    for k in keys:
        arrs = [d[k] for d in datasets]
        combined[k] = np.concatenate(arrs, axis=0)
    return combined


class SimOutput:
    def __init__(self, simulator):
        self.simulator = simulator

    def run(self):
        val = self.simulator()
        maps = val["results"]["maps"]
        if maps["mono_map"]:
            tau1 = tau2 = tau = maps.get("tau_map", maps.get("tau1_map"))
            alpha1 = 1.0
            efficiency = 1.0
        else:
            tau1, tau2 = maps["tau1_map"], maps["tau2_map"]
            tau = maps["tau_mean_map"]
            alpha1 = maps["alpha1_map"]
            efficiency = maps["fret_efficiency_map"]
        return {
            "decay": val["raw_data"]["decay"],
            "irf_original": val["raw_data"]["irf"],
            "tau1": tau1,
            "tau2": tau2,
            "tau": tau,
            "alpha1": alpha1,
            "photon_count": maps["photon_count_map"],
            "Efficiency": efficiency,
            "h_shift_jit_map": maps["h_shift_jit_map"],
            "h_shift_map": maps["h_shift_map"],
            "v_shift_map": maps["v_shift_map"],
        }


class SimOutputWithIRFOffset(SimOutput):
    def __init__(self, simulator, irf_1d):
        super().__init__(simulator)
        self.irf_1d = irf_1d

    def _compute_irf_offset(self):
        i_off = self.irf_1d
        i_off_sum = i_off.sum()
        if not np.isfinite(i_off_sum) or i_off_sum <= 0:
            raise ValueError(
                f"Invalid IRF: sum={i_off_sum}. IRF must be non-negative and non-zero."
            )
        i_off = i_off / i_off_sum
        return i_off

    def run(self):
        out = super().run()
        out["irf"] = self._compute_irf_offset()
        return out


class SimGenerator:
    """
    Wraps IRF shifting + a MacroSimulator/ContinuousSimulator/TCSPCSimulator/
    PhotonCountSimulator engine + SimOutputWithIRFOffset for a single config.
    Only accepts one config dict — raises if given a list/tuple of configs.

    Engine selection has two independent axes:

    - ``sensor_type`` (physics): ``config["sensor_type"]`` if present,
      otherwise the ``sensor_type`` constructor argument. ``"continuous"``
      samples from an intensity/ADC-scaled engine; ``"discrete"`` samples
      from a photon-by-photon TCSPC engine.
    - ``family`` (implementation): ``"separate"`` (default) uses
      :class:`ContinuousSimulator`/:class:`PhotonCountSimulator`;
      ``"combined"`` uses :class:`MacroSimulator`/:class:`TCSPCSimulator`.

    Parameters
    ----------
    family : str
        Which implementation family to sample from — "separate" (default)
        or "combined". See :data:`SIMULATOR_TYPES`.
    sensor_type : str
        Fallback "continuous"/"discrete" engine choice used only when
        ``config`` doesn't already set ``sensor_type``. Defaults to
        "discrete".
    """

    def __init__(
        self,
        irf_data,
        config,
        a_range=(-20, 100),
        b_range=(0, 10),
        pixel=(0, 0),
        family="separate",
        sensor_type="discrete",
    ):
        if isinstance(config, (list, tuple)):
            raise TypeError(
                "SimGenerator accepts exactly one config dict, not a "
                "list/tuple of configs. Instantiate one generator per config "
                "and combine their sampled outputs afterward if you need "
                "multiple configs."
            )
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config)}")
        if family not in SIMULATOR_TYPES:
            raise ValueError(
                f"family must be one of {list(SIMULATOR_TYPES)}, got {family!r}"
            )
        effective_sensor_type = str(config.get("sensor_type", sensor_type)).lower()
        if effective_sensor_type not in SIMULATOR_TYPES[family]:
            raise ValueError(
                f"sensor_type must be one of {list(SIMULATOR_TYPES[family])}, "
                f"got {effective_sensor_type!r}"
            )

        self.config = config
        self.simulator_cls = SIMULATOR_TYPES[family][effective_sensor_type]
        self.offset_gen = OffsetGen(
            irf_data, a_range=a_range, b_range=b_range, pixel=pixel
        )

    def simulate_once(self):
        """
        One pixel. ``h_shift`` (bins) is the delay relative to the base IRF: the applied
        IRF roll ``round(a)`` plus the jitter. ``irf`` already contains the roll and the
        baseline ``b``, so relative to it the solver's truth is ``h_shift_map`` (ns) and
        ``v_shift_map``.
        """
        irf_1d, a, b = self.offset_gen.sample()

        fli_simulator = self.simulator_cls(irf_data=irf_1d, **self.config)
        out = SimOutputWithIRFOffset(fli_simulator, irf_1d).run()
        out["h_shift_tof"] = a
        out["v_shift_bgp"] = b
        out["h_shift"] = round(a) + out["h_shift_jit_map"]
        return out


class WeightedConfigSimGenerator:
    """
    Wraps one :class:`SimGenerator` per config combination and, on every
    ``simulate_once()`` call, draws a fresh combination according to
    ``probs`` before delegating to it.

    This lets a single simulator built from this class (e.g. via
    ``bayesflow.make_simulator([lambda: gen.simulate_once()])``) produce
    draws sampled from a mixture of configs — such as the
    ``(configs, probs)`` pair returned by
    :meth:`~pyfli.data_cc.config_combinations.ConfigCombinationGenerator.combination_table` —
    with no change needed to code downstream that only ever calls
    ``simulate_once()``.

    Parameters
    ----------
    irf_data : np.ndarray
        Full IRF cube, forwarded to each per-combination :class:`SimGenerator`.
    configs : Sequence[dict]
        One config dict per combination.
    probs : Sequence[float]
        Sampling probability for each entry in ``configs`` (same order,
        same length). Renormalized if it doesn't already sum to 1.
    a_range, b_range, pixel, family, sensor_type
        Forwarded to every per-combination :class:`SimGenerator`.
    seed : int | None
        Seed for the combination-selection RNG (independent of each
        :class:`SimGenerator`'s own internal randomness).
    """

    def __init__(
        self,
        irf_data,
        configs,
        probs,
        a_range=(-20, 20),
        b_range=(0, 10),
        pixel=(0, 0),
        family="separate",
        sensor_type="discrete",
        seed=None,
    ):
        if len(configs) != len(probs):
            raise ValueError(
                f"configs and probs must be the same length, got "
                f"{len(configs)} and {len(probs)}"
            )
        self._generators = [
            SimGenerator(
                irf_data,
                cfg,
                a_range=a_range,
                b_range=b_range,
                pixel=pixel,
                family=family,
                sensor_type=sensor_type,
            )
            for cfg in configs
        ]
        probs = np.asarray(probs, dtype=float)
        self.probs = probs / probs.sum()
        self.rng = np.random.default_rng(seed)

    def simulate_once(self):
        """Picks one config combination per the weighted probabilities, then delegates to it."""
        idx = self.rng.choice(len(self._generators), p=self.probs)
        return self._generators[idx].simulate_once()


def make_simulator(simulate_fn, num_samples):
    """Run simulate_fn num_samples times and stack each output key along a new leading axis."""
    samples = [simulate_fn() for _ in range(num_samples)]
    keys = samples[0].keys()
    return {
        key: np.stack([np.asarray(s[key]) for s in samples], axis=0) for key in keys
    }


class BatchSimulator:
    """
    Run repeated FLI/FLIM simulations across parameter sets. The class is a convenience
    layer for generating batches of synthetic datasets for validation or model training.

    Unlike :func:`make_simulator`/:func:`concat_sim_data` (which batch the flattened
    per-sample dict produced by :class:`SimOutput`), these methods batch the raw nested
    ``{"raw_data": {...}, "results": {"maps": {...}, "TR_maps": {...}}}`` dict returned
    directly by a simulator's ``__call__`` (e.g. :class:`MacroSimulator`,
    :class:`~pyfli.simulator.separate.main_factory_gen.ContinuousSimulator`).
    """

    @staticmethod
    def _map_value_columns(samples: list, keys: Any) -> dict[Any, np.ndarray]:
        """
        Build one ``-1, 1``-shaped column per map key, filling ``np.nan`` for any
        sample that doesn't have that key. Samples aren't guaranteed to share the
        same map keys — e.g. a "mono" pixel from a ``mono_only_maps=True`` engine
        only has ``{tau_map, photon_count_map, mono_map}``, not ``tau1_map``/etc.
        """
        return {
            key: np.array(
                [s["results"]["maps"].get(key, np.nan) for s in samples]
            ).reshape(-1, 1)
            for key in keys
        }

    def sim_BI(self, sim_funcs: np.ndarray, num_list: int) -> Any:
        """
        Generates a simplified batch dictionary with specific parameters.
        Returns data as a dictionary of NumPy arrays.
        """
        samples = []
        for sim_func, n in zip(sim_funcs, num_list):
            samples.extend([sim_func() for _ in range(n)])

        if not samples:
            return {}

        # Wrapping each list in np.array for better performance and ML compatibility
        batch_data = {
            "decay": np.array([s["raw_data"]["decay"] for s in samples]),
            "irf": np.array([s["raw_data"]["irf"] for s in samples]),
            **self._map_value_columns(
                samples, ("tau1_map", "tau2_map", "alpha1_map", "photon_count_map")
            ),
        }
        return batch_data

    def generate_batch(self, sim_func_list: np.ndarray, num_list: int) -> Any:
        """
        Generate batch.

        Parameters
        ----------
        sim_func_list : np.ndarray
            Simulator functions used to generate a batch.
        num_list : int
            Number of samples generated for each simulator function.

        Returns
        -------
        Any
            Object produced by generate batch.
        """
        samples = []
        for sim_func, n in zip(sim_func_list, num_list):
            samples.extend([sim_func() for _ in range(n)])

        if not samples:
            return {}

        # Union of every sample's map keys, not just samples[0]'s — samples can
        # legitimately have different key sets (see _map_value_columns).
        map_keys = set()
        for s in samples:
            map_keys.update(s["results"]["maps"].keys())

        batch_data = {
            "raw_data": {
                "decay": np.stack([s["raw_data"]["decay"] for s in samples]),
                "irf": np.stack([s["raw_data"]["irf"] for s in samples]),
            },
            "results": {
                "maps": self._map_value_columns(samples, map_keys),
                "TR_maps": {
                    "fit_map": np.stack(
                        [s["results"]["TR_maps"]["fit_map"] for s in samples]
                    ),
                    "residual_map": np.stack(
                        [s["results"]["TR_maps"]["residual_map"] for s in samples]
                    ),
                },
            },
        }
        return batch_data

    def generate_batch2D(
        self, sim_funcs: np.ndarray, num_list: int, shape: tuple[int, ...] = (10, 10)
    ) -> Any:
        """
        Generate batch2 d.

        Parameters
        ----------
        sim_funcs : np.ndarray
            Simulator functions used to generate a two-dimensional batch.
        num_list : int
            Number of samples generated for each simulator function.
        shape : tuple[int, ...]
            Output shape requested for generated simulation batches.

        Returns
        -------
        Any
            Object produced by generate batch2d.
        """
        rows, cols = shape
        if sum(num_list) != rows * cols:
            raise ValueError(f"Sum of num_list must match shape product {rows * cols}")

        samples = []
        for sim_func, n in zip(sim_funcs, num_list):
            samples.extend([sim_func() for _ in range(n)])

        if not samples:
            return {}

        # Union of every sample's map keys, not just samples[0]'s — samples can
        # legitimately have different key sets (see _map_value_columns).
        map_keys = set()
        for s in samples:
            map_keys.update(s["results"]["maps"].keys())

        batch_data = {
            "raw_data": {
                "decay": np.stack([s["raw_data"]["decay"] for s in samples]).reshape(
                    rows, cols, -1
                ),
                "irf": np.stack([s["raw_data"]["irf"] for s in samples]).reshape(
                    rows, cols, -1
                ),
            },
            "results": {
                "maps": {
                    key: col.reshape(rows, cols)
                    for key, col in self._map_value_columns(samples, map_keys).items()
                },
                "TR_maps": {
                    "fit_map": np.stack(
                        [s["results"]["TR_maps"]["fit_map"] for s in samples]
                    ).reshape(rows, cols, -1),
                    "residual_map": np.stack(
                        [s["results"]["TR_maps"]["residual_map"] for s in samples]
                    ).reshape(rows, cols, -1),
                },
            },
        }
        return batch_data
