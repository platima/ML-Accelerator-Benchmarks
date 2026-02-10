# ML Accelerator Benchmarks

A benchmarking suite for ML-relevant matrix operations across the full spectrum of computing devices — from resource-constrained microcontrollers to single-board computers and desktops.

## What This Project Does

This suite runs standardised matrix multiplication and element-wise operation benchmarks using NumPy-compatible libraries across three Python runtimes:

- **CPython** (with NumPy) — for Linux SBCs and desktops
- **MicroPython** (with ulab) — for microcontrollers like RP2350, ESP32
- **CircuitPython** (with ulab) — for microcontrollers like RP2040, RP2350

Results are output as structured JSON for easy comparison across wildly different hardware.

## Key Features

- Automatic hardware detection and configuration
- Memory-aware matrix size optimisation
- Multi-core awareness
- Temperature and power monitoring (where available)
- Standardised performance metrics
- JSON output format for parsing and comparison

## Quick Start

See the [Getting Started](getting-started.md) guide for installation and usage instructions.

## Supported Hardware

See the [Hardware Compatibility](benchmarks/hardware.md) page for the full list of tested platforms.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
