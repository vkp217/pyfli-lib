# FAQ

**Which acquisition systems does `pyfli` support?**
ICCD, gated-SPAD (SwissSPAD2/SwissSPAD3), TCSPC, and more are natively supported. `DataOperations` auto-detects the format from the input file.
If a format you need isn't listed, feel free to open a discussion — the developers check in periodically and add support where possible.

**Do I need a GPU?**
No. All analytical fitting methods (NLSF, MLE, RLD, Laguerre, phasor) run on CPU. A GPU is only used by the optional deep-learning inference backends — install the extra with `pip install "pyfli-lib[gpu]"` if you need it.

**Why is the package `pyfli-lib` but I `import pyfli`?**
The name `pyfli` was already taken on PyPI, so the distribution is published as `pyfli-lib` while the importable module keeps its natural name, `pyfli`.

**Which fitting method should I use?**
It depends on your data. NLSF is the standard general-purpose choice; MLE is better suited to low-photon-count data; RLD is the fastest option for real-time or high-frame-rate processing. Phasor analysis is model-free and useful for quick visual species separation without fitting a model at all.

**How do I cite `pyfli`?**
See the {doc}`citation` page for the full BibTeX entry, or cite the repository directly: <https://github.com/vkp217/pyfli-lib>.

**Where do I report a bug or request a feature?**
On the [GitHub issue tracker](https://github.com/vkp217/pyfli-lib/issues).

**Where can I get help?**
Open an issue on GitHub, or email [support@pyfli.org](mailto:support@pyfli.org) or [pyfli4lifetime@gmail.com](mailto:pyfli4lifetime@gmail.com).
