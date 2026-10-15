"""End-to-end calibration: experimental 3D decay cube -> pyfli simulator config."""

import os
from dataclasses import asdict, dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import beta as beta_dist

from pyfli import logging

from ..sim_calibrator import FLIValidator
from ..sim_workflow import SimGenerator
from .config import SIM_FAMILY, CalibrationInputs
from .coverage import coverage_fractions, ensure_coverage
from .data_io import (
    ExperimentalData,
    ReferenceMaps,
    load_experimental_data,
    load_reference_maps,
)
from .expected_decay import DecayForwardModel
from .irf_estimation import estimate_irf, split_background
from .optimizer import (
    CalibrationState,
    ObjectiveContext,
    evaluate,
    refresh_photon_prior,
    run_calibration_rounds,
    stage_parameter_names,
)
from .pixel_selection import select_signal_pixels
from .prior_estimation import BetaPrior, PriorSet, lifetime_priors, quick_pixel_fits
from .sim_bridge import build_sim_config, make_generator, save_calibration
from .summary_stats import (
    StatsComparator,
    background_bins,
    compute_stats,
    inactive_bins,
)


@dataclass
class CalibrationResult:
    config: dict
    sim_generator_kwargs: dict
    state: CalibrationState
    mask: np.ndarray
    selection_report: dict
    validation: dict
    history: pd.DataFrame
    output_dir: str
    files: dict = field(default_factory=dict)
    best_fit: dict = field(default_factory=dict)

    def make_generator(self) -> SimGenerator:
        kw = self.sim_generator_kwargs
        return make_generator(
            self.state.irf.irf_cube(),
            self.config,
            kw["a_range"],
            kw["b_range"],
            inactive_bins=kw.get("inactive_bins"),
        )


class DetectorSimCalibrator:
    def __init__(self, inputs: CalibrationInputs) -> None:
        self.inp = inputs
        self.rng = np.random.default_rng(inputs.seed)
        data_path = os.path.abspath(inputs.data_path.rstrip("/"))
        base = os.path.splitext(os.path.basename(data_path))[0]
        self.output_dir = inputs.output_dir or os.path.join(
            os.path.dirname(data_path),
            f"{base}_sim_calibration_bin{inputs.bin_radius}_{inputs.detector_type}_{inputs.model_type}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ steps
    def _initial_lifetimes(
        self, fm, data: ExperimentalData, mask, irf_est, dcr, ref: ReferenceMaps | None
    ):
        """Parameter values used to initialise the Beta priors, plus where they came from."""
        inp = self.inp
        if ref is not None:
            sel = mask if ref.aligned_with(data.spatial_shape) else None
            values = {
                k: ref.values(k, sel)
                for k in ("tau1", "tau2", "alpha1", "efficiency", "tau")
            }
            key = "tau2" if inp.is_bi else "tau"
            if len(values[key]) >= 50:
                if sel is None:
                    logging.warning(
                        "Reference maps are not pixel-aligned with the data: using all their valid pixels"
                    )
                if not inp.is_bi and len(values["tau"]) == 0:
                    values["tau"] = values["tau2"]
                return values, f"reference:{ref.source}"
            logging.warning(
                "Reference maps have too few valid pixels: falling back to quick fits"
            )
        coords = np.argwhere(mask)
        pick = coords[
            self.rng.choice(
                len(coords), min(inp.n_fit_pixels, len(coords)), replace=False
            )
        ]
        decays = data.decay[pick[:, 0], pick[:, 1]]
        pf = irf_est.pooled_fit
        init = {
            "tau2": pf["tau_long_ns"],
            "efficiency": 1 - pf["tau_short_ns"] / pf["tau_long_ns"],
            "alpha1": pf["frac_short"],
            "tau": pf["frac_short"] * pf["tau_short_ns"]
            + (1 - pf["frac_short"]) * pf["tau_long_ns"],
        }
        irf_norm = fm.with_baseline(irf_est.effective_irf(), irf_est.b_max / 2)
        values = quick_pixel_fits(
            fm, decays, irf_norm, np.full(len(decays), dcr), inp.is_bi, init
        )
        return values, "quick_fit"

    def run(self) -> CalibrationResult:
        inp = self.inp
        data = load_experimental_data(
            inp.data_path, inp.irf_path, inp.bg_path, inp.mask_path, inp.bin_radius
        )
        T = data.n_bins
        adc_ceiling = (
            (2**inp.bit_depth - 1)
            if (inp.detector_type == "continuous" and inp.bit_depth)
            else None
        )

        # 1. sample pixels only
        mask, sel_report = select_signal_pixels(
            data.decay, data.mask, inp.selection, adc_ceiling
        )
        if mask.sum() < 20:
            raise RuntimeError(
                "Too few sample pixels selected; see the selection report / relax the criteria"
            )

        # 2. experimental statistics on a random subset of the sample pixels
        coords = np.argwhere(mask)
        order = self.rng.permutation(len(coords))
        n_cal = min(inp.n_exp_pixels, len(coords))
        sub = coords[order[:n_cal]]
        held_out = coords[order[n_cal : n_cal + inp.n_exp_pixels]]
        exp_decays = data.decay[sub[:, 0], sub[:, 1]].astype(np.float64)
        inactive = inactive_bins(data.decay.sum((0, 1)))
        sel_report["inactive_bins"] = inactive.tolist()
        if len(inactive):
            logging.info(
                f"{len(inactive)} time bins never record a count (outside the acquisition window): "
                "they are zeroed in the simulated decays"
            )
        bg_bins = background_bins(exp_decays.sum(0))
        exp_stats = compute_stats(exp_decays, bg_bins)

        # pyfli's jitter shifts every decay by U{-2..2} bins (edge SD ~1.4 bins). Data whose
        # rising edges are tighter than that cannot have been produced with jitter on.
        edge_sd = float(
            np.subtract(*np.nanpercentile(exp_stats["edge"], [75, 25])) / 1.349
        )
        if inp.jitter == "auto":
            inp.jitter = edge_sd >= 1.0
            logging.info(
                f"Rising-edge SD {edge_sd:.2f} bins -> simulator jitter {'on' if inp.jitter else 'off'}"
            )
        sel_report["edge_sd_bins"] = edge_sd

        # 3. IRF (experimental template or simulated EMG) + pooled lifetimes / background
        fm = DecayForwardModel(
            T, inp.laser_period_ns, inp.detector_type, inp.jitter, inactive
        )
        irf_est = estimate_irf(fm, data.decay, mask, data.irf, inp.irf_mode)
        # dark counts: pre-edge level = dcr + slope * counts across pixels (the slope is the
        # signal-proportional previous-pulse tail), so the intercept is the dark-count rate
        pre = exp_decays[:, bg_bins].mean(-1)
        slope, intercept = np.polyfit(exp_stats["counts"], pre, 1)
        if fm.circular:
            dcr0 = max(split_background(irf_est, fm, bg_bins, slope, intercept), 1e-4)
        else:
            dcr0 = max(intercept, irf_est.pooled_fit["dcr_per_bin_per_pixel"], 1e-4)
        sel_report["dcr_regression"] = {
            "intercept": float(intercept),
            "slope": float(slope),
        }

        # 4. initial lifetime priors from reference maps or quick fits
        ref = (
            load_reference_maps(
                inp.reference_path, inp.reference_source, inp.reference_alpha
            )
            if inp.reference_path
            else None
        )
        values, prior_source = self._initial_lifetimes(
            fm, data, mask, irf_est, dcr0, ref
        )
        tau2_p, eff_p, a1_p, moments = lifetime_priors(
            values, inp.is_bi, inp.laser_period_ns, inp.prior_inflation
        )

        exp_counts = data.decay[mask].sum(-1).astype(float)
        exp_peaks = uniform_filter1d(exp_decays, 3, axis=-1).max(-1) - dcr0
        ctx = ObjectiveContext(
            fm=fm,
            detector_type=inp.detector_type,
            laser_freq_mhz=inp.laser_freq_mhz,
            jitter=inp.jitter,
            comparator=StatsComparator(exp_stats),
            bg_bins=bg_bins,
            exp_counts=exp_counts,
            exp_peaks=exp_peaks,
            data_max=float(exp_decays.max()),
            bit_depth=inp.bit_depth,
            prior_inflation=inp.prior_inflation,
            n_sim=inp.n_sim_pixels,
            seed=inp.seed,
            reference_moments=moments if ref is not None else {},
            reference_weight=inp.reference_weight if ref is not None else 0.0,
            inactive_bins=inactive,
        )
        priors = PriorSet(
            tau2_p,
            eff_p,
            a1_p,
            BetaPrior(2, 2),
            (0, 1),
            16,
            inp.mono_fraction,
            prior_source,
            moments,
        )
        if inp.detector_type == "continuous":
            # pyfli scales the configured dcr by bit / 8; store the config value
            state = refresh_photon_prior(CalibrationState(irf_est, priors, dcr0), ctx)
            state = replace(state, dcr=dcr0 / (state.priors.bit / 8.0))
            ctx = replace(ctx, bit_depth=state.priors.bit)
        else:
            state = refresh_photon_prior(CalibrationState(irf_est, priors, dcr0), ctx)
        init_state = state
        init_dist, init_parts, *_ = evaluate(state, ctx)
        logging.info(f"Initial distance {init_dist:.4f}: {init_parts}")

        # 5. simulation-based optimisation
        nuisance, prior_names = stage_parameter_names(
            state,
            ctx,
            inp.optimize_irf,
            inp.optimize_noise,
            inp.optimize_priors,
            inp.is_bi,
        )
        state, hist = run_calibration_rounds(
            state,
            ctx,
            nuisance,
            prior_names,
            inp.n_bank_pixels,
            inp.bank_rounds,
            inp.max_evals_stage1,
            inp.max_evals_stage2,
        )
        history = pd.DataFrame(hist)
        best_fit = state

        cov_seed = inp.seed + 4243
        coverage_trace = []
        if inp.target_coverage > 0:
            state, coverage_trace, cov_cal, _, cov_sim_stats = ensure_coverage(
                state,
                ctx,
                exp_stats,
                exp_counts,
                inp.target_coverage,
                inp.n_coverage_pixels,
                inp.max_coverage_iter,
                seed=cov_seed,
            )
        else:
            _, _, _, cov_sim_stats = evaluate(
                state, ctx, n=inp.n_coverage_pixels, seed=cov_seed
            )
            cov_cal = coverage_fractions(exp_stats, cov_sim_stats)
        cov_held = None
        if len(held_out) >= 100:
            held_stats = compute_stats(
                data.decay[held_out[:, 0], held_out[:, 1]].astype(np.float64), bg_bins
            )
            cov_held = coverage_fractions(held_stats, cov_sim_stats)
        _, _, _, best_sim_stats = evaluate(
            best_fit, ctx, n=inp.n_coverage_pixels, seed=cov_seed
        )
        cov_best = coverage_fractions(exp_stats, best_sim_stats)
        logging.info(
            f"Coverage of the experiment: best fit {cov_best} -> final {cov_cal}; held-out pixels {cov_held}"
        )

        # 6. independent validation (fresh random numbers, more pixels)
        val_seed = inp.seed + 7919
        dist, parts, sim, sim_stats = evaluate(
            state, ctx, n=inp.n_validation_pixels, seed=val_seed
        )
        best_val, best_val_parts, *_ = evaluate(
            best_fit, ctx, n=inp.n_validation_pixels, seed=val_seed
        )
        init_val, init_val_parts, *_ = evaluate(
            init_state, ctx, n=inp.n_validation_pixels, seed=val_seed
        )
        validator = FLIValidator(method="calibration", threshold=1)
        vm = validator.run_comprehensive_test(
            {"raw_data": {"decay": sim["decay"]}}, exp_decays
        )
        validation = {
            "distance_initial": init_val,
            "distance_initial_parts": init_val_parts,
            "distance_best_fit": best_val,
            "distance_best_fit_parts": best_val_parts,
            "distance_final": dist,
            "distance_final_parts": parts,
            "coverage_target": inp.target_coverage,
            "coverage_best_fit": cov_best,
            "coverage_final": cov_cal,
            "coverage_held_out": cov_held,
            "n_held_out_pixels": len(held_out),
            "coverage_trace": coverage_trace,
            "fli_validator": {
                k: float(vm[k])
                for k in (
                    "cosine_similarity",
                    "kl_divergence",
                    "ks_p_value",
                    "hist_intersection",
                )
            },
            "n_validation_pixels": inp.n_validation_pixels,
        }
        logging.info(
            f"Validation: distance {init_val:.3f} (initial) -> {best_val:.3f} (best fit) -> {dist:.3f} "
            f"(covering); {validation['fli_validator']}"
        )

        # 7. save
        config = build_sim_config(
            state.priors, inp.detector_type, state.dcr, inp.laser_freq_mhz, inp.jitter
        )
        kw = {
            "a_range": [round(v, 3) for v in state.a_range()],
            "b_range": [0.0, float(state.irf.b_max)],
            "pixel": [0, 0],
            "family": SIM_FAMILY,
            "sensor_type": inp.detector_type,
            "inactive_bins": inactive.tolist(),
        }
        result = CalibrationResult(
            config, kw, state, mask, sel_report, validation, history, self.output_dir
        )
        result.best_fit = {
            "config": build_sim_config(
                best_fit.priors,
                inp.detector_type,
                best_fit.dcr,
                inp.laser_freq_mhz,
                inp.jitter,
            ),
            "sim_generator_kwargs": {
                **kw,
                "a_range": [round(v, 3) for v in best_fit.a_range()],
                "b_range": [0.0, float(best_fit.irf.b_max)],
            },
            "priors": best_fit.priors.summary(),
            "dcr_config": best_fit.dcr,
        }
        self._save(
            result, data, exp_decays, exp_stats, sim, sim_stats, values, init_state
        )
        return result

    # ----------------------------------------------------------------- output
    def _save(
        self,
        res: CalibrationResult,
        data,
        exp_decays,
        exp_stats,
        sim,
        sim_stats,
        values,
        init_state,
    ) -> None:
        out = self.output_dir
        inp_dict = asdict(self.inp)
        files = {
            "irf": "calibrated_irf.npy",
            "mask": "signal_mask.npy",
            "history": "optimization_history.csv",
            "validation_sample": "validation_simulated_pixels.npz",
        }
        np.save(os.path.join(out, files["irf"]), res.state.irf.irf_cube())
        np.save(os.path.join(out, files["mask"]), res.mask)
        res.history.to_csv(os.path.join(out, files["history"]), index=False)
        np.savez_compressed(os.path.join(out, files["validation_sample"]), **sim)
        st = res.state
        payload = {
            "inputs": inp_dict,
            "config": res.config,
            "sim_generator_kwargs": res.sim_generator_kwargs,
            "irf": {
                "source": st.irf.source,
                "file": files["irf"],
                "params": st.irf.params,
                "a_range": list(st.a_range()),
                "b_range": list(st.b_range()),
                "pooled_fit": st.irf.pooled_fit,
            },
            "priors": st.priors.summary(),
            "priors_initial": init_state.priors.summary(),
            "best_fit": res.best_fit,
            "dcr_config": st.dcr,
            "photon_scale": st.photon_scale,
            "pixel_selection": res.selection_report,
            "validation": res.validation,
            "usage": "from pyfli.simulator.calibration import load_calibrated_generator; "
            "gen = load_calibrated_generator('<this file>'); gen.simulate_once()",
        }
        files["config"] = "calibrated_sim_config.json"
        save_calibration(os.path.join(out, files["config"]), payload)
        files.update(
            self._plots(res, data, exp_decays, exp_stats, sim, sim_stats, values)
        )
        res.files = {k: os.path.join(out, v) for k, v in files.items()}
        logging.info(f"Calibration saved to {out}")

    def _plots(self, res, data, exp_decays, exp_stats, sim, sim_stats, values) -> dict:
        out, st = self.output_dir, res.state
        files = {}

        # pixel selection
        inten = data.decay.sum(-1)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        im = ax[0].imshow(np.log10(inten + 1), cmap="magma")
        ax[0].set_title("log10 intensity")
        fig.colorbar(im, ax=ax[0])
        ax[1].imshow(res.mask, cmap="gray")
        ax[1].set_title(
            f"sample mask ({res.selection_report['mode']}): {int(res.mask.sum())} px"
        )
        bins = np.linspace(0, np.log10(inten.max() + 1), 120)
        ax[2].hist(
            np.log10(inten[~res.mask] + 1),
            bins,
            alpha=0.6,
            label="rejected (background / noise)",
        )
        ax[2].hist(np.log10(inten[res.mask] + 1), bins, alpha=0.6, label="sample")
        ax[2].set(xlabel="log10 counts / pixel", yscale="log")
        ax[2].legend()
        for a in ax[:2]:
            a.axis("off")
        fig.tight_layout()
        files["plot_selection"] = "pixel_selection.png"
        fig.savefig(os.path.join(out, files["plot_selection"]), dpi=130)
        plt.close(fig)

        # simulated vs experimental decays
        fig, ax = plt.subplots(2, 3, figsize=(17, 9))
        ax[0, 0].semilogy(exp_stats["mean_decay"], label="experimental (sample pixels)")
        ax[0, 0].semilogy(sim_stats["mean_decay"], "--", label="calibrated simulation")
        irf_n = st.irf.effective_irf()
        ax[0, 0].semilogy(
            irf_n / irf_n.max() * exp_stats["mean_decay"].max(),
            ":",
            label=f"IRF ({st.irf.source})",
        )
        ax[0, 0].set(
            title="pooled decay shape",
            xlabel="time bin",
            ylim=(
                exp_stats["mean_decay"].max() * 1e-4,
                exp_stats["mean_decay"].max() * 3,
            ),
        )
        ax[0, 0].legend()
        b = np.linspace(
            min(exp_stats["log_counts"].min(), sim_stats["log_counts"].min()),
            max(exp_stats["log_counts"].max(), sim_stats["log_counts"].max()),
            70,
        )
        ax[0, 1].hist(
            exp_stats["log_counts"], b, density=True, alpha=0.55, label="experimental"
        )
        ax[0, 1].hist(
            sim_stats["log_counts"], b, density=True, alpha=0.55, label="simulated"
        )
        ax[0, 1].set(title="photons / pixel", xlabel="log10 counts")
        ax[0, 1].legend()
        e1, s1 = exp_stats["phasor1"], sim_stats["phasor1"]
        ax[0, 2].hexbin(
            s1[:, 0], s1[:, 1], gridsize=70, cmap="Blues", mincnt=1, bins="log"
        )
        k = slice(None, None, max(1, len(e1) // 4000))
        ax[0, 2].scatter(e1[k, 0], e1[k, 1], s=1, c="r", alpha=0.3)
        ax[0, 2].set(
            title="raw phasor (h=1): simulated (blue) vs experimental (red)",
            xlabel="G",
            ylabel="S",
        )
        for a, key, title in (
            (ax[1, 0], "edge", "rising-edge position (bins)"),
            (ax[1, 1], "bg_frac", "pre-edge background / mean level"),
        ):
            e, s = (
                exp_stats[key][np.isfinite(exp_stats[key])],
                sim_stats[key][np.isfinite(sim_stats[key])],
            )
            bb = np.linspace(
                np.percentile(np.r_[e, s], 0.5), np.percentile(np.r_[e, s], 99.5), 70
            )
            a.hist(e, bb, density=True, alpha=0.55, label="experimental")
            a.hist(s, bb, density=True, alpha=0.55, label="simulated")
            a.set_title(title)
            a.legend()
        e2, s2 = exp_stats["phasor2"], sim_stats["phasor2"]
        ax[1, 2].hexbin(
            s2[:, 0], s2[:, 1], gridsize=70, cmap="Blues", mincnt=1, bins="log"
        )
        ax[1, 2].scatter(e2[k, 0], e2[k, 1], s=1, c="r", alpha=0.3)
        ax[1, 2].set(title="raw phasor (h=2)", xlabel="G", ylabel="S")
        v = res.validation
        fig.suptitle(
            f"{self.inp.detector_type} / {self.inp.model_type}: distance {v['distance_initial']:.2f} -> {v['distance_final']:.2f}"
        )
        fig.tight_layout()
        files["plot_validation"] = "sim_vs_experiment.png"
        fig.savefig(os.path.join(out, files["plot_validation"]), dpi=130)
        plt.close(fig)

        # priors vs initial values
        names = [
            (
                "tau2",
                "tau2" if self.inp.is_bi else "tau",
                "tau2 (ns)" if self.inp.is_bi else "tau (ns)",
            )
        ]
        if self.inp.is_bi:
            names += [
                ("efficiency", "efficiency", "efficiency E = 1 - tau1/tau2"),
                ("alpha1", "alpha1", "A1 (photon fraction of tau1)"),
            ]
        fig, ax = plt.subplots(1, len(names) + 1, figsize=(4.8 * (len(names) + 1), 4))
        ax = np.atleast_1d(ax)
        for a, (pname, vkey, title) in zip(ax, names):
            p = getattr(st.priors, pname)
            x = np.linspace(p.offset, p.offset + p.scale, 300)
            vals = values.get(vkey, np.array([]))
            vals = vals[np.isfinite(vals)]
            if len(vals):
                a.hist(
                    vals,
                    60,
                    range=(p.offset, p.offset + p.scale),
                    density=True,
                    alpha=0.45,
                    label=f"initial ({st.priors.source})",
                )
            sim_key = {"tau2": "tau2", "efficiency": "Efficiency", "alpha1": "alpha1"}[
                pname
            ]
            a.hist(
                sim[sim_key],
                60,
                range=(p.offset, p.offset + p.scale),
                density=True,
                histtype="step",
                lw=1.5,
                label="calibrated simulation",
            )
            a.plot(
                x,
                beta_dist.pdf((x - p.offset) / p.scale, p.a, p.b) / p.scale,
                "k",
                lw=2,
                label=f"Beta({p.a:.2f}, {p.b:.2f})",
            )
            a.set_title(title)
            a.legend(fontsize=8)
        ax[-1].hist(sim["photon_count"], 60, density=True, alpha=0.6)
        ax[-1].set_title(
            "photon prior: "
            + (
                "expected photons (0.01 x n_cycles)"
                if self.inp.detector_type == "discrete"
                else "peak A (ADC)"
            )
        )
        fig.tight_layout()
        files["plot_priors"] = "priors.png"
        fig.savefig(os.path.join(out, files["plot_priors"]), dpi=130)
        plt.close(fig)

        if len(res.history):
            fig, a = plt.subplots(figsize=(8, 3.5))
            for stage, g in res.history.groupby("stage", sort=False):
                a.plot(g.index, g["objective"].cummin(), label=f"{stage} (best so far)")
                a.plot(g.index, g["objective"], ".", ms=2, alpha=0.4)
            a.set(xlabel="evaluation", ylabel="objective", yscale="log")
            a.legend()
            fig.tight_layout()
            files["plot_history"] = "optimization_history.png"
            fig.savefig(os.path.join(out, files["plot_history"]), dpi=130)
            plt.close(fig)
        return files
