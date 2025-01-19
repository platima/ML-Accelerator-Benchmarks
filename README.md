# Universal ML Accelerator Benchmark Suite 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MicroPython](https://img.shields.io/badge/micropython-1.19+-yellow.svg)](https://micropython.org/)
[![CircuitPython](https://img.shields.io/badge/circuitpython-8.2+-blue.svg)](https://circuitpython.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmarking suite for ML accelerators across the entire spectrum of computing devices - from powerful edge AI accelerators to resource-constrained microcontrollers.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Supported Hardware](#supported-hardware)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmark Metrics](#benchmark-metrics)
- [Example Results](#example-results)
- [Future Development](#future-development)
- [Contributing](#contributing)
- [License](#license)

## Overview
This benchmark suite provides standardized performance metrics for matrix operations commonly used in ML workloads. It automatically detects hardware capabilities and maximizes matrix sizes based on available memory.

## Features
- Automatic hardware detection and configuration
- Memory-aware matrix size optimization
- Multi-core awareness
- Temperature and power monitoring (where available)
- Standardized performance metrics
- JSON output format for easy parsing and comparison

## Supported Hardware
Currently tested and supported platforms:
- RP2040 (Raspberry Pi Pico) - MicroPython + ulab (Piromoni)
- RP2350 (Raspberry Pi Pico W 2) - MicroPython + ulab (Piromoni), CircuitPython
- ESP32-P4 - CircuitPython

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/platima/ml-accelerator-benchmark.git
   ```
2. Choose the appropriate version for your platform:
   - `micropython-benchmark.py` for MicroPython devices
   - `circuitpython-benchmark.py` for CircuitPython devices

## Usage
1. Upload the appropriate benchmark file to your device
2. Run the benchmark:
   ```python
   import benchmark
   benchmark = UniversalBenchmark()
   results = benchmark.run()
   ```

## Benchmark Metrics
The benchmark provides several key metrics:
- **Matrix Operations**: Maximum supported matrix size and performance
- **Memory Usage**: Total available and used memory
- **Performance Metrics**:
  - Raw inference time
  - Operations per second
  - Normalized score (ops/second/MHz)
  - Theoretical power (accounting for cores and frequency)
- **Hardware Monitoring**:
  - Temperature (where available)
  - Power usage (where available)

## Example Results
```json
{
    "_meta": {
        "Source version": "0.2",
        "Source code": "https://github.com/platima/ml-accelerator-benchmark/blob/main/micropython-benchmark.py",
        "Source repo": "https://github.com/platima/ml-accelerator-benchmark",
        "Test date": "2025-01-19",
        "Tester": "Platima"
    },
    "device": {
        "board_type": "rp2350",
        "cpu_freq_mhz": 150.000,
        "num_cores": 2,
        "temp_sensor": true,
        "power_sensor": true
    },
    "performance": {
        "channels": 3,
        "array_size": 134,
        "memory_total": 480160,
        "memory_used": 288096,
        "min_inference_ms": 1759.000,
        "max_inference_ms": 1770.000,
        "avg_inference_ms": 1760.100,
        "throughput_fps": 0.568
    },
    "benchmark": {
        "total_ops": 7332828,
        "ops_per_second": 4166143,
        "normalized_score": 27774.290,
        "theoretical_power": 8332287
    }
}
```

## Future Development
The future planned version of this benchmark suite can be found in the [future-development branch](https://github.com/platima/ml-accelerator-benchmark/tree/future-development). Note that this branch contains work-in-progress features that are currently untested.

## Contributing
Contributions are welcome! Key areas for improvement include:
- Additional hardware support
- Improved detection methods
- New benchmark metrics
- Documentation improvements
- Bug fixes and optimizations

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.