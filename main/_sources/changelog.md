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
  fetch("https://api.github.com/repos/vkp217/pyfli-pkg/issues?state=open&per_page=20")
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
      a.href = "https://github.com/vkp217/pyfli-pkg/issues";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "View issues on GitHub";
      p.appendChild(a);
      container.appendChild(p);
    });
})();
</script>
```

[View all issues on GitHub](https://github.com/vkp217/pyfli-pkg/issues)

## Release history

<!-- UNRELEASED 0.1.20: not shown on the site. On release, delete this line and the closing line below, and set the date.

### [0.1.20](https://pypi.org/project/pyfli-lib/0.1.20/) — YYYY-MM-DD

```{card}
:class-card: sd-bg-success

**Features added:**
- `weighting` option for NLSF (`"irls"` default, `"none"`, `"neyman"`), for MLE (`"poisson"` default, `"pearson"`, `"neyman"`, `"none"`) and for GPU NLSF
- Iteratively reweighted least squares (IRLS): NLSF now gives the same result as Poisson MLE for photon-count data
- Gate-integrated forward model: `h_shift` is a true, smooth sub-gate onset shift (zero before onset, earlier curve for negative shifts); CPU, GPU and reconstruction use the same model
- `CVPlot`: log-space and count-weighted fitting of the ideal `1/√N` trend.
- New examples: NLSF & MLE fitting for mono- and bi-exponential data
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
```

```{card}
:class-card: sd-bg-warning

**Bugs Fixed:**
- Fixed a sign error in the truncated-window phasor locus
```

END OF UNRELEASED 0.1.20 -->

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
