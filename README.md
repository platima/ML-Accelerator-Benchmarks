<img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=platima.mlbenchmark" height="20" />

# Universal ML Accelerator Benchmark Suite 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MicroPython](https://img.shields.io/badge/micropython-1.19+-yellow.svg)](https://micropython.org/)
[![CircuitPython](https://img.shields.io/badge/circuitpython-8.2+-blue.svg)](https://circuitpython.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A benchmarking suite for ML-relevant matrix operations across the spectrum of computing devices — from desktops to resource-constrained microcontrollers. Automatically detects hardware, maximises matrix sizes based on available memory, and outputs standardised JSON results.

## Features

- **Three runtimes**: CPython + NumPy, MicroPython + ulab, CircuitPython + ulab
- Automatic board and CPU detection
- Memory-aware matrix-size probing with 0.9× safety factor
- Multi-core awareness
- Temperature monitoring (where available)
- Standardised JSON output with schema validation
- Utilities for analysis, visualisation, and comparison

## Supported Hardware

| Device | Architecture | Runtime |
|--------|-------------|---------|
| Intel i7-14700K | x86-64 | CPython |
| Luckfox Omni3576 | ARM64 (RK3576) | CPython |
| SpacemiT MUSE Pi Pro | RISC-V 64 | CPython |
| Luckfox Pico Zero | ARM32 (RV1103) | CPython |
| Milk-V Duo 256 | RISC-V 64 (CV1812H) | CPython |
| Raspberry Pi 4 / 5 | ARM64 | CPython |
| RP2350 (Pico 2) | ARM Cortex-M33 | MicroPython, CircuitPython |
| RP2040 (Pico) | ARM Cortex-M0+ | MicroPython, CircuitPython |
| ESP32-S3 (LX7) | Xtensa LX7 | MicroPython |
| ESP32-C6 | RISC-V 32 | MicroPython |
| ESP32 (LX6) | Xtensa LX6 | MicroPython |
| ESP32-P4 | RISC-V | CircuitPython |

## Quick Start

### Desktop / SBC (CPython)

```bash
pip install numpy psutil
python benchmarks/python-benchmark.py
```

### MicroPython

Upload `benchmarks/micropython-benchmark.py` to the device and run via REPL.

### CircuitPython

Copy `benchmarks/circuitpython-benchmark.py` to the device as `code.py` or run via REPL.

### Analysing Results

```bash
# Validate all result files
python -m utils.validate_results

# Print comparison report
python -m utils.benchmark_analyzer
```

## Result Format

Every benchmark outputs a JSON object with four sections:

| Section | Contents |
|---------|----------|
| `_meta` | Source version, date, tester, firmware URL |
| `device` | Board type, CPU frequency, core count, sensors |
| `performance` | Array size, inference times, memory, temperature |
| `benchmark` | Total ops, ops/sec, normalised score, theoretical power |

See [Understanding Results](https://platima.github.io/ML-Accelerator-Benchmarks/benchmarks/results/) for full field definitions and the schema at `results/results-schema.json`.

## Utilities

The `utils/` package provides:

| Module | Purpose |
|--------|---------|
| `benchmark_analyzer` | Load results, rank devices, generate text reports |
| `visualization` | matplotlib comparison charts (CPython only) |
| `micropython_viz` | ASCII bar charts for the REPL |
| `results_handler` | Save / load / list result files |
| `hardware_detect` | Platform, CPU, and accelerator detection |
| `validate_results` | Schema + consistency validation |

## Documentation

Full documentation is available at [platima.github.io/ML-Accelerator-Benchmarks](https://platima.github.io/ML-Accelerator-Benchmarks/).

## Future Development

The planned next-generation benchmark suite is in the [future-development branch](https://github.com/platima/ML-Accelerator-Benchmarks/tree/future-development). That branch contains untested work-in-progress features.

## Contributing

Contributions are welcome — hardware results, bug fixes, new board detection, and documentation improvements. See the [Contributing Guide](https://platima.github.io/ML-Accelerator-Benchmarks/contributing/) for conventions.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
