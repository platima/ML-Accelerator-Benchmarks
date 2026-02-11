# ML Accelerator Benchmarks

A benchmarking suite for ML-relevant matrix operations across the full spectrum of computing devices — from resource-constrained microcontrollers to single-board computers and desktops.

## What This Project Does

This suite runs standardised matrix multiplication and element-wise operation benchmarks using NumPy-compatible libraries across three Python runtimes:

- **CPython** (with NumPy) — for Linux SBCs and desktops
- **MicroPython** (with ulab) — for microcontrollers like RP2350, ESP32
- **CircuitPython** (with ulab) — for microcontrollers like RP2040, RP2350, ESP32-P4

Results are output as structured JSON for easy comparison across wildly different hardware.

## Key Features

- Automatic hardware and board detection
- Memory-aware matrix-size probing with 0.9× safety factor
- Multi-core awareness
- Temperature monitoring (where available)
- Standardised JSON output with schema validation
- Utilities for analysis, charts, and comparison reports

## Quick Links

| | |
|---|---|
| [Getting Started](getting-started.md) | Installation and first run |
| [Running Benchmarks](benchmarks/running.md) | Per-runtime instructions |
| [Hardware Compatibility](benchmarks/hardware.md) | Supported devices |
| [Understanding Results](benchmarks/results.md) | JSON format and metrics |
| [Testing Checklist](benchmarks/testing.md) | Per-device verification steps |
| [API Reference](api/index.md) | Utils package documentation |

## License

Apache 2.0 — see the [LICENSE](https://github.com/platima/ML-Accelerator-Benchmarks/blob/main/LICENSE) file for details.
