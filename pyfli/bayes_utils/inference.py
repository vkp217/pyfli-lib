"""
model_inference.py

Class-based wrapper around the Bi direct-inference pipeline:
  - loads a trained bi-/mono-exponential Keras model (pass `custom_objects`
    at construction time if the checkpoint references custom architecture
    classes -- this module no longer defines any of its own)
  - runs spatial-patch, mask-aware inference: fully masked-out patches are
    skipped entirely (no model call), and within a partially-masked patch
    only the masked-in pixels are sent to the model
  - restitches per-pixel medians/MADs into output maps (masked-out pixels
    are left at 0), and keeps the full per-pixel posterior sample stack
    (`output_samples`) so downstream code can compute other statistics
    from the same posterior draws without re-running inference
  - optionally saves outputs via a `saver` object
  - optionally runs DetailedRecon.reconstruct on the outputs
  - optionally visualizes results via DataViewer

Usage
-----

.. code-block:: python

    from model_inference import BiPipeline

    pipeline = BiPipeline(
        model_type="bi-exponential",
        model_weights="/mnt/e/.../biexpon/model.keras",
        patch_size=(128, 128),
    )

    results = pipeline.run(
        decay=binned_decay,
        irf=binned_irf,
        mask=b_bool_mask,
        freq=freq[1],
        saver=saver,
        save_outputs=True,
        compute_detailed=True,
        visualize=True,
    )

    output_maps          = results["output_maps"]
    output_uncertainties = results["output_uncertainties"]
    output_samples        = results["output_samples"]   # (H, W, NUM_SAMPLES) per key
    bi_detailed           = results["detailed"]          # bi_bi or bi_mono
"""

import os
from typing import ClassVar

import keras
import numpy as np
from scipy.stats import median_abs_deviation

from pyfli.data_vnp import ColorProcessor, DataViewer
from pyfli.reconstruction import DetailedRecon


class BiPipeline:
    """Encapsulates the Bi direct-inference pipeline for FLI decay maps."""

    #: keys produced per model type
    MODEL_KEYS: ClassVar[dict[str, list[str]]] = {
        "bi-exponential": ["tau1", "tau2", "alpha1"],
        "mono-exponential": ["tau"],
    }

    def __init__(
        self,
        model_type,
        model_weights,
        patch_size=(128, 128),
        num_samples=100,
        batch_size=1024,
        custom_objects=None,
    ):
        """
        Parameters
        ----------
        model_type : str
            "bi-exponential" or "mono-exponential".
        model_weights : str
            Path to the trained Keras checkpoint for `model_type`.
        patch_size : tuple(int, int)
            (PATCH_SIZE1, PATCH_SIZE2) spatial patch grid used during inference.
        num_samples : int
            Number of posterior samples drawn per pixel during model.sample().
        batch_size : int
            Batch size passed to model.sample().
        custom_objects : dict, optional
            Passed straight through to `keras.saving.load_model` -- required
            only if the checkpoint references custom (de)serializable
            architecture classes not built into Keras.
        """
        if model_type not in self.MODEL_KEYS:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type!r}")
        if not model_weights:
            raise ValueError("model_weights must be a path to a Keras checkpoint")

        self.model_type = model_type
        self.model_weights = model_weights
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.custom_objects = custom_objects
        self.keys = self.MODEL_KEYS[model_type]

        self.loaded_model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        """Load the Keras model checkpoint from `self.model_weights`.

        Pass `custom_objects` at construction time if the checkpoint
        references custom architecture classes -- this module no longer
        defines any inline.
        """
        if not os.path.exists(self.model_weights):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_weights}")

        self.loaded_model = keras.saving.load_model(
            self.model_weights, custom_objects=self.custom_objects
        )
        return self.loaded_model

    # ------------------------------------------------------------------
    # Inference (spatial-patch + mask-aware)
    # ------------------------------------------------------------------
    def run_inference(self, decay, irf, mask):
        """
        Run spatial-patch inference, matching the confirmed-working reference
        implementation's input path exactly: each patch is sent to the model
        as a full, fixed-size batch (same shape/order the model was
        validated against), never a masked/subsetted batch.

        Mask awareness is applied in two places only:
          1. A patch is skipped entirely -- no model call at all -- if every
             pixel in it is masked out.
          2. After the model call, masked-out pixels in that patch are zeroed
             back out in the output maps (post-hoc), so the model itself
             never sees a reduced or variable-size batch.

        This deliberately avoids sending a variable-size, masked-down subset
        of pixels into model.sample() -- an earlier version of this method
        did that as a compute-saving optimization, but it changed the batch
        composition/size the model saw per patch and produced near-uniform,
        wrong per-pixel outputs. Skipping fully-empty patches is safe (it
        doesn't change what any *processed* patch's batch looks like);
        subsetting pixels within a partially-masked patch is not.

        Parameters
        ----------
        decay : np.ndarray, shape (H, W, N_BINS)
        irf   : np.ndarray, shape (N_BINS,) shared across every pixel, or
            (H, W, N_BINS) per-pixel -- same convention as
            ParamToDecay/DetailedRecon.
        mask  : np.ndarray, shape (H, W), bool-like

        Returns
        -------
        output_maps : dict[str, np.ndarray] each (H, W) -- per-pixel posterior
            median for each key.
        output_uncertainties : dict[str, np.ndarray] each (H, W) -- per-pixel
            median absolute deviation (MAD) of the posterior samples, via
            scipy.stats.median_abs_deviation's default scale=1.0. This is the
            *raw* MAD, not scaled to be std-comparable: for a roughly Gaussian
            posterior it runs ~0.6745x the equivalent standard deviation (use
            scale='normal' at the call site below if a 1-sigma-comparable
            value is ever needed instead).
        output_samples : dict[str, np.ndarray] each (H, W, NUM_SAMPLES) --
            the raw per-pixel posterior draws model.sample() produced, kept
            around so other statistics can be computed later directly from
            them instead of re-running inference.
        """
        if self.loaded_model is None:
            self.load_model()

        img_size1, img_size2 = decay.shape[0], decay.shape[1]
        n_bins = decay.shape[-1]
        patch_size1, patch_size2 = self.patch_size

        irf = np.asarray(irf)
        if irf.ndim not in (1, 3):
            raise ValueError(
                f"irf must be 1-D (N_BINS,) or 3-D (H, W, N_BINS); got shape {irf.shape}"
            )

        # ceil division so trailing rows/cols aren't silently dropped when
        # img_size isn't an exact multiple of patch_size
        num_patches_side1 = int(np.ceil(img_size1 / patch_size1))
        num_patches_side2 = int(np.ceil(img_size2 / patch_size2))

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (img_size1, img_size2):
            raise ValueError(
                f"mask shape {mask.shape} does not match "
                f"decay spatial shape {(img_size1, img_size2)}"
            )

        output_maps = {key: np.zeros((img_size1, img_size2)) for key in self.keys}
        output_uncertainties = {
            key: np.zeros((img_size1, img_size2)) for key in self.keys
        }
        output_samples = {
            key: np.zeros((img_size1, img_size2, self.num_samples)) for key in self.keys
        }

        k = 0
        n_patches_total = num_patches_side1 * num_patches_side2
        n_patches_skipped = 0

        for i in range(num_patches_side1):
            for j in range(num_patches_side2):
                r_start, r_end = i * patch_size1, min((i + 1) * patch_size1, img_size1)
                c_start, c_end = j * patch_size2, min((j + 1) * patch_size2, img_size2)

                patch_mask = mask[r_start:r_end, c_start:c_end]
                if not patch_mask.any():
                    n_patches_skipped += 1
                    continue  # fully masked-out patch -- no model call at all

                k += 1
                print(
                    f"processing patch: {k} (row {i}, col {j}), "
                    f"{patch_mask.sum()}/{patch_mask.size} pixels valid"
                )

                patch_h, patch_w = patch_mask.shape

                # Full, fixed-size patch batch -- same shape/order the model
                # expects, matching the confirmed-working reference exactly.
                # No pixel subsetting here.
                patch_decay = decay[r_start:r_end, c_start:c_end, :].reshape(-1, n_bins)
                if irf.ndim == 3:
                    patch_irf = irf[r_start:r_end, c_start:c_end, :].reshape(
                        -1, irf.shape[-1]
                    )
                else:
                    patch_irf = np.broadcast_to(
                        irf, (patch_decay.shape[0], irf.shape[-1])
                    )

                patch_conditions = {
                    "decay": patch_decay,
                    "irf": patch_irf,
                }
                out = self.loaded_model.sample(
                    conditions=patch_conditions,
                    num_samples=self.num_samples,
                    batch_size=self.batch_size,
                )

                for key in self.keys:
                    samples_2d = out[key].squeeze(axis=-1)  # (n_pixels, num_samples)
                    val = np.median(samples_2d, axis=1)
                    unc = median_abs_deviation(samples_2d, axis=1)

                    patch_vals = val.reshape(patch_h, patch_w)
                    patch_unc = unc.reshape(patch_h, patch_w)
                    patch_samples = samples_2d.reshape(
                        patch_h, patch_w, self.num_samples
                    )

                    # Post-hoc masking: zero out masked-out pixels only after
                    # the model has produced output for the full patch.
                    patch_vals = np.where(patch_mask, patch_vals, 0.0)
                    patch_unc = np.where(patch_mask, patch_unc, 0.0)
                    patch_samples = np.where(patch_mask[..., None], patch_samples, 0.0)

                    output_maps[key][r_start:r_end, c_start:c_end] = patch_vals
                    output_uncertainties[key][r_start:r_end, c_start:c_end] = patch_unc
                    output_samples[key][r_start:r_end, c_start:c_end, :] = patch_samples

        print(
            f"Restitching done ({n_patches_skipped}/{n_patches_total} patches fully masked out and skipped)"
        )
        first_key = next(iter(output_maps))
        print(f"Final Map Shape: {output_maps[first_key].shape}")
        print(
            f"Final Samples Shape: {output_samples[first_key].shape} and number of keys are: {len(output_samples.keys())}"
        )

        return output_maps, output_uncertainties, output_samples

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_outputs(
        self, saver, output_maps, output_uncertainties, output_samples=None, tag=None
    ):
        """Save output_maps, output_uncertainties, and (if given) output_samples
        via the provided saver object."""
        tag = tag or self.model_type
        saver.save_npy(f"Bi Direct_Output_{tag}", output_maps)
        saver.log("The Bi Direct_Output saved")
        saver.save_npy(f"Bi output_uncertainties_{tag}", output_uncertainties)
        saver.log("The Bi output_uncertainties saved")
        if output_samples is not None:
            saver.save_npy(f"Bi output_samples_{tag}", output_samples)
            saver.log("The Bi output_samples saved")

    def save_detailed(self, saver, detailed, name=None):
        """Save the DetailedRecon.reconstruct dict via the provided saver object."""
        name = (
            name
            or f"Bi {'bi' if self.model_type == 'bi-exponential' else 'mono'}_Output"
        )
        saver.save_npy(name, detailed)
        saver.log(f"The {name} saved")

    # ------------------------------------------------------------------
    # Detailed results (DetailedRecon.reconstruct)
    # ------------------------------------------------------------------
    def compute_detailed(
        self,
        output_maps,
        freq,
        decay,
        irf,
        shift_method="zero_pad",
        h_shift=None,
        v_shift=None,
        excitation="bin_start",
        gate_width=None,
    ):
        """
        Run DetailedRecon.reconstruct on the output maps for the configured model_type,
        with the forward model `shift_method` (and `excitation`, `gate_width`) and
        optional onset ``h_shift`` (ns) and background ``v_shift`` (scalars or (H, W)
        maps; ``output_maps`` keys win).

        Returns the raw results dict (bi_bi or bi_mono equivalent).
        """
        if self.model_type == "bi-exponential":
            cdr_params = {
                "tau1_map": output_maps["tau1"],
                "tau2_map": output_maps["tau2"],
                "alpha1_map": output_maps["alpha1"],
            }
            data_name = "Bi_bi"
        elif self.model_type == "mono-exponential":
            cdr_params = {"tau_map": output_maps["tau"]}
            data_name = "Bi_mono"
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {self.model_type!r}")
        ref_shape = np.shape(next(iter(cdr_params.values())))
        for key, value in (("h_shift", h_shift), ("v_shift", v_shift)):
            value = output_maps.get(key, value)
            if value is not None:
                cdr_params[f"{key}_map"] = np.broadcast_to(
                    np.asarray(value, dtype=float), ref_shape
                ).copy()

        reconstructor = DetailedRecon(
            freq,
            irf,
            binned_decay=decay,
            shift_method=shift_method,
            excitation=excitation,
            gate_width=gate_width,
        )
        detailed = reconstructor.reconstruct(
            cdr_params, self.model_type, data_name=data_name
        )

        print(detailed["results"]["maps"].keys())
        print(detailed["results"]["TR_maps"].keys())
        return detailed

    # ------------------------------------------------------------------
    # Visualization (only meaningful for bi-exponential outputs)
    # ------------------------------------------------------------------
    @staticmethod
    def default_cmap():
        """Lazily build the default colormap (jet with lowest value pinned to zero)."""
        return ColorProcessor().lowest_zero("jet")

    def visualize(self, saver, output_maps, output_uncertainties, cmap=None):
        """
        Display parameter maps and their uncertainties using DataViewer.
        Only applicable when model_type == "bi-exponential" (alpha1/tau1/tau2).
        `cmap` defaults to `default_cmap()` (jet_m) when not provided.

        No pixel-coordinate/decay-curve argument here: every array plotted
        (output_maps/output_uncertainties) is a (H, W) 2-D map, and
        DataViewer.display_data only adds its extra decay-curve panel when a
        3-D (H, W, T) array is present in data_list -- so a `coord` would
        never draw anything, just reserve a blank column.
        """
        if self.model_type != "bi-exponential":
            raise ValueError(
                "visualize() currently only supports bi-exponential outputs"
            )

        if cmap is None:
            cmap = self.default_cmap()

        data_list1 = [output_maps["alpha1"], output_maps["tau1"], output_maps["tau2"]]
        data_names1 = ["alpha1_map_Bi", "tau1_map_Bi", "tau2_map_Bi"]
        v_ranges1 = [(0, 1), (0, 2), (0, 2)]

        data_list = [
            output_uncertainties["alpha1"],
            output_uncertainties["tau1"],
            output_uncertainties["tau2"],
        ]
        data_names = ["alpha1_map_Bi_UN", "tau1_map_Bi_UN", "tau2_map_Bi_UN"]
        v_ranges = [(0, 1), (0, 1), (0, 1)]

        cmaps = [cmap, cmap, cmap]
        cols = len(data_list)

        DataViewer(save_path=saver.save_dir, fig_name="Bi_parameters").display_data(
            data_list1,
            structure=(1, cols),
            data_names=data_names1,
            cmaps=cmaps,
            v_ranges=v_ranges1,
            figsize=None,
            normalize=False,
            yscale="linear",
        )
        DataViewer(save_path=saver.save_dir, fig_name="Bi_parameters_UN").display_data(
            data_list,
            structure=(1, cols),
            data_names=data_names,
            cmaps=cmaps,
            v_ranges=v_ranges,
            figsize=None,
            normalize=False,
            yscale="linear",
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(
        self,
        decay,
        irf,
        mask,
        freq,
        saver=None,
        save_outputs=True,
        compute_detailed=True,
        visualize=False,
        cmap=None,
    ):
        """
        Run the full pipeline: load model -> inference -> (save) -> (detailed) -> (visualize).

        `cmap` is optional; if omitted and `visualize=True`, defaults to
        `default_cmap()` (jet with lowest value pinned to zero, i.e. jet_m).

        Returns
        -------
        dict with keys: "output_maps", "output_uncertainties", "output_samples", "detailed"
        ("detailed" is None if compute_detailed=False)
        """
        if self.loaded_model is None:
            self.load_model()

        output_maps, output_uncertainties, output_samples = self.run_inference(
            decay, irf, mask
        )

        if save_outputs:
            if saver is None:
                raise ValueError("save_outputs=True requires a `saver` object")
            self.save_outputs(saver, output_maps, output_uncertainties, output_samples)

        detailed = None
        if compute_detailed:
            detailed = self.compute_detailed(output_maps, freq, decay, irf)
            if save_outputs:
                if saver is None:
                    raise ValueError("save_outputs=True requires a `saver` object")
                self.save_detailed(saver, detailed)

        if visualize:
            if saver is None:
                raise ValueError("visualize=True requires a `saver` object")
            if cmap is None:
                cmap = self.default_cmap()
            self.visualize(saver, output_maps, output_uncertainties, cmap)

        return {
            "output_maps": output_maps,
            "output_uncertainties": output_uncertainties,
            "output_samples": output_samples,
            "detailed": detailed,
        }
