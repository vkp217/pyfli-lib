# Quickstart

This example loads a decay dataset together with its instrument response function (IRF), the starting point for any `pyfli` analysis.

Even though the package is installed as `pyfli-lib`, you import it as `pyfli` in your scripts:

```python
from pyfli import DataOperations

loader = DataOperations(
    data_path="experimental_data.sdt",
    irf_path="instrument_data.txt",
    bg_path="background_data.tif",
    mask_path="background_data.png",
)
decay_data = loader.load_data()
irf_data = loader.load_irf()
```

`DataOperations` auto-detects the acquisition format from the file extension (`.sdt` for SPCImage/TCSPC, among others), and applies background subtraction using `bg_path` by default.

## Next steps

- Browse the {doc}`API Reference <api/index>` for the full set of loaders, analytical fitting methods (NLSF, MLE, RLD, Laguerre), and the phasor-analysis toolkit.
- See {doc}`citation` if you use `pyfli` in published research.
