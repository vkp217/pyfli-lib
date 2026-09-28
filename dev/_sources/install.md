# Install

## From PyPI

Install the stable release directly from PyPI:

```bash
pip install pyfli-lib
```

Even though the distribution is named `pyfli-lib`, you always `import pyfli` in your scripts.

## Optional extras

For users requiring GPU-accelerated fitting, install the optional tensor/AI dependencies:

```bash
pip install "pyfli-lib[gpu]"
```

For deep-learning inference backends that require TensorFlow:

```bash
pip install "pyfli-lib[tf]"
```

## From source

To work on `pyfli` itself, clone the repository and install it in editable mode with the development extras (`pytest`, `black`, `pre-commit`):

```bash
git clone https://github.com/vkp217/pyfli-pkg.git
cd pyfli-pkg
pip install -e ".[dev]"
```

Then set up the pre-commit hooks (ruff lint + format on every commit):

```bash
pre-commit install
```

## From GitHub (dev branch, no clone)

Install the latest `dev` branch directly with `pip`, without cloning the repository:

```bash
pip install "git+https://github.com/vkp217/pyfli-pkg.git@dev"
```

## Requirements

`pyfli` requires **Python 3.11+**. Core dependencies (NumPy, SciPy, scikit-image, OpenCV, PyTorch, PySide6, and others) are installed automatically — see `pyproject.toml` for the full list.

## Verify the install

```bash
python -c "import pyfli; print('pyfli is installed correctly')"
```
