# Contributing

Contributions are welcome! `pyfli` is developed on GitHub at [vkp217/pyfli-pkg](https://github.com/vkp217/pyfli-pkg).

## Development setup

```bash
git clone https://github.com/vkp217/pyfli-pkg.git
cd pyfli-pkg
pip install -e ".[dev]"
pre-commit install
```

## Making a change

1. Fork the repository and create a branch for your change.
2. Make your changes, with tests where applicable.
3. Open a pull request against the `dev` branch.

## Running the test suite

```bash
python -m pytest
```

Tests live in `tests/` and run automatically on every pull request via GitHub Actions (`.github/workflows/tests.yml`).

## Code style

Linting and formatting are enforced with [ruff](https://docs.astral.sh/ruff/) through `pre-commit` (`.github/workflows/code-style.yml` runs the same checks in CI):

```bash
pre-commit run --all-files
```

## Building the docs

The documentation you're reading is built with Sphinx from `docsrc/`:

```bash
cd docsrc
make html
```

Open `docsrc/_build/html/index.html` in a browser to preview your changes.

## Reporting issues

Found a bug or have a feature request? Please open an issue on the [issue tracker](https://github.com/vkp217/pyfli-pkg/issues).
