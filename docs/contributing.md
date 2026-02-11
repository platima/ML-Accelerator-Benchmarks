# Contributing

Contributions are welcome! This guide covers the development workflow and conventions used in this project.

## Development Setup

1. Clone the repository and switch to the **develop** branch:

    ```bash
    git clone https://github.com/platima/ML-Accelerator-Benchmarks.git
    cd ML-Accelerator-Benchmarks
    git checkout develop
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-docs.txt
    ```

3. Preview documentation locally:

    ```bash
    mkdocs serve
    ```

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `chore:` — maintenance tasks (deps, CI, etc.)
- `refactor:` — code restructuring without behaviour change
- `test:` — adding or updating tests

## Versioning

This project follows [Semantic Versioning](https://semver.org/) (semver):

- **MAJOR** — incompatible changes to result format or benchmark methodology
- **MINOR** — new features, new hardware support, phase completions
- **PATCH** — bug fixes, documentation updates, minor improvements

## Branching Workflow

All development happens on the **develop** branch. When a release is ready, `develop` is merged into `main` and tagged.

1. Create a feature branch from `develop`
2. Make your changes
3. Submit a pull request **against `develop`** (not `main`)

## Submitting Results

If you've run benchmarks on hardware not yet in the `results/` directory:

1. Run the appropriate benchmark script
2. Save the JSON output to `results/` using the naming convention: `YYYY-MM-DD Board Runtime.json`
3. Validate your result against the schema (see below)
4. Submit a pull request against `develop`

## Schema Validation

Result JSON files should conform to `results/results-schema.json`. You can validate with:

```bash
python -m utils validate
```

## Code Style

- Use Australian English in comments and documentation (e.g. "colour", "normalised", "optimisation")
- Use `.format()` string formatting in MicroPython and CircuitPython scripts (not f-strings, for broader compatibility)
- Use Google-style docstrings for Python utilities

## Areas for Contribution

- Additional hardware support and board detection
- Improved detection methods
- New benchmark metrics
- Documentation improvements
- Bug fixes and optimisations
- Result submissions from new hardware
