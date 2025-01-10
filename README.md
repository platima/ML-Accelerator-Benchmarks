# Universal ML Accelerator Benchmark Suite 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MicroPython](https://img.shields.io/badge/micropython-1.19+-yellow.svg)](https://micropython.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmarking suite for ML accelerators across the entire spectrum of computing devices - from powerful edge AI accelerators to resource-constrained microcontrollers. Compare performance and capabilities across different platforms.

## 🎯 Features

- Universal compatibility from SoCs to MCUs
- Automatic hardware capability detection
- Standardized benchmark results format
- Matrix operation benchmarks
- Model inference benchmarks
- Memory benchmarks

## 🖥️ Implementation Status

| Platform | Status | Runner | Models |
|----------|---------|---------|---------|
| RP2040/RP2350 | ✅ | micropython-ulab-runner.py | Matrix operations |
| RV1103/RV1106 | ✅ | rknn-runner.py | MobileNetV2, YOLOv5s |
| RK3588 | ✅ | rknn-runner.py | MobileNetV2, YOLOv5s/v8n |
| CV1800B/SG2002 | ✅ | cvitek-runner.py | MobileNetV2, YOLOv5s |
| RK3399 | 🚧 | neon-simd-runner.py | TBD |
| BCM2711 | 🚧 | videocore-runner.py | TBD |
| Other CPU/GPU | ✅ | python-runner.py | Matrix operations |

## 📊 Results Format

All benchmark results follow a standardized JSON schema (see `schema/results.json`). Example structure:

```json
{
  "device": {
    "name": "Device Name",
    "type": "SBC/SOC/MCU",
    "processor": {
      "name": "Processor Name",
      "architecture": "Architecture",
      "frequency_mhz": 1000
    },
    "accelerator": {
      "name": "Accelerator Name",
      "type": "NPU/TPU/GPU/NONE",
      "compute_capability": "0.5 TOPS"
    }
  },
  "benchmarks": {
    "matrix_ops": {
      "max_size": 1024,
      "performance": {
        "inference_ms": {"min": 4.5, "max": 5.2, "avg": 4.8},
        "throughput": {"operations_per_second": 208.3}
      }
    },
    "models": [
      {
        "name": "mobilenetv2",
        "format": "onnx",
        "performance": {
          "inference_ms": {"min": 15.2, "max": 16.8, "avg": 15.9},
          "throughput": {"fps": 62.8}
        }
      }
    ]
  }
}
```

See `examples/results.json` for a complete example.

## 🔧 Installation

### Standard Python Version
```bash
git clone https://github.com/yourusername/universal-ml-benchmark
cd universal-ml-benchmark
pip install -r requirements.txt
```

### MicroPython Version
1. Install MicroPython on your board
2. Copy required files:
   - `micropython-ulab-runner.py`
   - `hardware_detect.py`
   - `memory_benchmark.py`

### Platform-Specific Requirements
- RKNN: `pip install rknn-toolkit2`
- CVITEK: Install TPU-MLIR SDK
- Others: See platform documentation

## 📈 Usage

### Running All Available Benchmarks
```bash
python benchmark.py
```

### Running Specific Benchmarks
```bash
# RKNN Platforms
python rknn-runner.py

# MicroPython Platforms
python micropython-ulab-runner.py

# CVITEK Platforms
python cvitek-runner.py
```

## 🤝 Contributing

Contributions are welcome! Key areas:
- Additional hardware support
- Improved detection methods
- New benchmark metrics
- Documentation improvements
- Bug fixes

## TODO List

- [ ] VideoCore implementation for BCM2711
- [ ] NEON SIMD implementation for RK3399
- [ ] Additional model support
- [ ] Expanded memory benchmarks
- [ ] Result visualization tools

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
