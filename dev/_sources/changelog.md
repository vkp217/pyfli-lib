# Changelog

Version history for `pyfli`, and currently open issues pulled live from
GitHub. Version numbers and dates below follow the releases published on
[PyPI](https://pypi.org/project/pyfli-lib/).

## Open issues

```{raw} html
<div id="gh-issues-widget">
  <p><em>Loading open issues from GitHub…</em></p>
</div>
<script>
(function () {
  var container = document.getElementById("gh-issues-widget");
  fetch("https://api.github.com/repos/vkp217/pyfli-lib/issues?state=open&per_page=20")
    .then(function (r) {
      if (!r.ok) { throw new Error("GitHub API error: " + r.status); }
      return r.json();
    })
    .then(function (items) {
      var issues = items.filter(function (i) { return !i.pull_request; });
      container.innerHTML = "";
      if (issues.length === 0) {
        var p = document.createElement("p");
        p.textContent = "No open issues right now.";
        container.appendChild(p);
        return;
      }
      var list = document.createElement("ul");
      issues.forEach(function (issue) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = issue.html_url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = "#" + issue.number + " " + issue.title;
        li.appendChild(a);
        list.appendChild(li);
      });
      container.appendChild(list);
    })
    .catch(function (err) {
      container.innerHTML = "";
      var p = document.createElement("p");
      p.textContent = "Could not load issues from GitHub (" + err.message + "). ";
      var a = document.createElement("a");
      a.href = "https://github.com/vkp217/pyfli-lib/issues";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "View issues on GitHub";
      p.appendChild(a);
      container.appendChild(p);
    });
})();
</script>
```

[View all issues on GitHub](https://github.com/vkp217/pyfli-lib/issues)

## Release history

### [0.1.20](https://pypi.org/project/pyfli-lib/0.1.20/) — 2026-10-14

```{card}
:class-card: sd-bg-success

**Features added:**
- `weighting` option for NLSF (`"irls"` default, `"none"`, `"neyman"`), for MLE (`"poisson"` default, `"pearson"`, `"neyman"`, `"none"`) and for GPU NLSF
- Iteratively reweighted least squares (IRLS): NLSF now gives the same result as Poisson MLE for photon-count data
- Gate-integrated forward model: `h_shift` is a true, smooth sub-gate onset shift (zero before onset, earlier curve for negative shifts); CPU, GPU and reconstruction use the same model
- `CVPlot`: log-space and count-weighted fitting of the ideal `1/√N` trend.
- New examples: NLSF & MLE fitting for mono- and bi-exponential data
- PicoQuant `.ptu` file loading (`load_ptu_file`, frames integrated, channel selectable; adds the `ptufile` dependency)
- Robust (sandwich) NLSF uncertainties for `weighting="none"`, and column-scaled Jacobian for numerically stable errors
- Global fitter passes `weighting` / `max_iter` to the super-pixel fit
- New metrics in `pyfli.solver`: `poisson_deviance`, `expected_poisson_deviance`, `reduced_poisson_deviance`, `pearson_chi_square`, `compute_pearson_stats`
- `pyfli.plot_style`: darker colour palettes and legends outside the axes across analysis and `data_vnp` plots
- `PhasorAdditionalPlots.multidata_phasor_plot`
- `pyfli.simulator.calibration`: calibrate the simulator (priors, IRF, detector noise) to experimental data
- `pyfli.analysis.precision`: Cramér-Rao bounds, fit-free lifetimes and Monte Carlo precision studies
- `phasor_clustering`: phasor PCA, clustering and LDA (scikit-learn)
- `phasor_lifetime`: normal and apparent phasor lifetimes (`PhasorAnalyzer.compute_normal_lifetime`, `compute_apparent_lifetimes`)
- `ModelSelector` / `ModelSelectorPlotter`: per-pixel mono- vs bi-exponential model selection (F-test + AICc) and figures
- `pyfli.bayes_utils`: posterior MAD (`unc`), credible-interval (`trustworthiness_CrI`) and goodness-of-fit (`trustworthiness_gof`) trustworthiness maps
```

```{card}
:class-card: sd-bg-danger

**Changes that affect results:**
- `photon_count_map` is now the number of photons (previously photons × gate width, about 20× smaller)
- NLSF default weighting changed from Neyman to IRLS, so NLSF lifetimes differ from earlier versions (`weighting="neyman"` reproduces them)
- `use_weights` is deprecated; use `weighting`
- `max_iter` now limits the MLE optimizer
- Reported `Red.χ²` / `reduced_chi2_map` is now the Poisson deviance divided by its expected value (≈ 1 for a correct fit at any photon count; `chi2_map` is the deviance); the former Pearson values are kept as `pearson_chi2_map` / `pearson_reduced_chi2_map`
- Removed `pyfli.phasor.phasorSEPL` and its top-level re-exports; use `pyfli.phasor.phasorS.MonoLocus`
- License changed from CC BY-NC-ND 4.0 to AGPL-3.0
- Fit initial guesses: amplitude is now the photon count (was the peak value) and `h_shift` starts at half a gate (was 0)
- `SimGenerator` default IRF shift range `a_range` is now (-20, 20) (was (-20, 100))
- Removed `pyfli.simulator.calibration_engine` (`FLICalibrator`) and `pyfli.simulator.sim_stat_test`; use `pyfli.simulator.calibration`
```

```{card}
:class-card: sd-bg-warning

**Bugs Fixed:**
- Fixed a sign error in the truncated-window phasor locus
- Fixed NLSF parameter uncertainties not being reordered with τ1/τ2 when bi-exponential lifetimes were swapped
- Fixed phasor math: acquisition loci, colour bar, harmonic lifetimes, two-component fractions and per-bin kernels
```

### [0.1.19](https://pypi.org/project/pyfli-lib/0.1.19/) — 2026-08-31

```{card}
:class-card: sd-bg-info sd-font-weight-bold
:text-align: center

⚡ A major refactor in the library
```

```{card}
:class-card: sd-bg-success

**Features added:**
- Expanded simulator functionality
- Additional image generation features
- Detailed Phasor example
- Synthetic IRF simulator
- Time-of-flight offset
- Updated IRF aligner
- KDE support in Phasor analysis
- Gate signal and truncated signal Phasor
- Elaborated plotting and processed-data comparison against underlying properties
- Elaborated examples of Simulator and Phasor analysis
```

```{card}
:class-card: sd-bg-warning

**Bugs Fixed:**
- Continuous and discrete simulators being dropped
```

### [0.1.18](https://pypi.org/project/pyfli-lib/0.1.18/) — 2026-06-25

```{card}
:class-card: sd-bg-success

**Features added:**
- *(add notes for this release)*
```

```{card}
:class-card: sd-bg-warning

**Bugs Fixed:**
- *(add notes for this release)*
```

### [0.1.17](https://pypi.org/project/pyfli-lib/0.1.17/) — 2026-06-03

```{card}
:class-card: sd-bg-success

**Features added:**
- Simulator
- Detailed Phasor Method
- NLSF, MLE, and plotting methods
```

```{card}
:class-card: sd-bg-warning

**Bugs Fixed:**
- Subplotting placement error
```
