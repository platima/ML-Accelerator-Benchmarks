# Universal ML Accelerator Benchmark Suite 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MicroPython](https://img.shields.io/badge/micropython-1.19+-yellow.svg)](https://micropython.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmarking suite for ML accelerators across the entire spectrum of computing devices - from powerful edge AI accelerators to resource-constrained microcontrollers.

The future planned version or state - for when I have time - can be found [here](https://github.com/platima/ml-accelerator-benchmark/tree/future-development) but most of it is currently untested or heavily WIP.

## 
- RP2350/RP2040 (MicroPython + ulab)

## Example Results Format
```json
{
    {
        "device": {
            "cpu_freq_mhz": 150,
            "board_type": "rp2350",
            "temp_sensor": true,
            "power_sensor": true
        },
        "performance": {
            "max_inference_ms": 1770,
            "array_size": 134,
            "min_inference_ms": 1759,
            "memory_total": 480160,
            "throughput_fps": 0.568,
            "avg_temperature": 26.529,
            "memory_used": 288096,
            "max_temperature": 27.044,
            "avg_inference_ms": 1760.100,
            "channels": 3
        }
    }
}
```

## Contributing

Contributions welcome! Key areas:
- Additional hardware support
- Improved detection methods
- New benchmark metrics
- Documentation improvements

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
