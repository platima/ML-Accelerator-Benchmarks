# Universal ML Accelerator Benchmark Suite 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MicroPython](https://img.shields.io/badge/micropython-1.19+-yellow.svg)](https://micropython.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

------------------------------------------------------------------------------------------------------

🚨 **CURRENTLY HEAVILY WORK-IN-PROGRESS AND I THINK RIGHT NOW MOST OF IT WON'T WORK AS I RE-FACTOR IT ALL** 🚨

[Here is the code for the MicroPython version](https://github.com/platima/ml-accelerator-benchmark/blob/816c4b2387aaf601e4a23f96c9798f3b306a766d/micropython-benchmark.py) I used in the [Waveshare RP2350-Plus](https://youtu.be/bAN2rE3MwOs) video.

First test results from RP2040 and RP2350 using that code [can be found here](results/2025-01-10%20RP2040%20RP2350%20Platima.json)

------------------------------------------------------------------------------------------------------

A comprehensive benchmarking suite for ML accelerators across the entire spectrum of computing devices - from powerful edge AI accelerators to resource-constrained microcontrollers.

## Supported Hardware

### TPU/NPU (Full Support)
- Rockchip RV1103/RV1106 (RKNN - 0.5 TOPS)
- Rockchip RK3588S (RKNN - 6 TOPS)
- CVITEK CV1800B (TPU-MLIR - 0.5 TOPS)
- CVITEK SG2002 (TPU-MLIR)

### Other Platforms
- RK3399 (NEON SIMD)
- BCM2711 (VideoCore VI)
- SpacemiT K1
- RP2350/RP2040 (MicroPython + ulab)

## Supported Models
- MobileNetV2 (Classification)
- YOLOv5s (Object Detection)

## Installation

### Prerequisites

For Rockchip RKNN platforms:
```bash
# Install RKNN Toolkit 2
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2
pip install -r packages/requirements_cp38-1.6.0.txt  # Choose version based on Python
pip install packages/rknn_toolkit2-1.6.0+xxx-cp38-cp38-linux_x86_64.whl
```

For CVITEK platforms:
```bash
# Setup Docker environment
git clone https://github.com/sophgo/tpu-mlir.git
cd tpu-mlir
source ./docker/build.sh
source ./tpu-mlir/envsetup.sh

# Install TPU-MLIR Python package
pip install cvi-toolkit
```

### Main Installation
```bash
git clone https://github.com/yourusername/universal-ml-benchmark
cd universal-ml-benchmark
pip install -r requirements.txt
```

## Usage

### Running Complete Benchmark Suite
```bash
python main-benchmark.py
```

### Running TPU/NPU Specific Benchmarks
```bash
python tpu-benchmark.py \
  --models mobilenetv2,yolov5s \
  --num-runs 100
```

### Example Results Format
```json
{
  "device": {
    "name": "RV1106",
    "type": "SBC",
    "processor": {
      "name": "RV1106",
      "architecture": "ARM64",
      "frequency_mhz": 1008
    },
    "accelerator": {
      "name": "RKNN NPU",
      "type": "NPU",
      "compute_capability": "0.5 TOPS"
    }
  },
  "benchmarks": {
    "models": [
      {
        "name": "mobilenetv2",
        "format": "onnx",
        "performance": {
          "inference_ms": {
            "min": 15.2,
            "max": 16.8,
            "avg": 15.9
          },
          "throughput": {
            "fps": 62.8
          }
        }
      }
    ]
  }
}
```

## Platform-Specific Setup

### Rockchip RV1106
1. Install RKNN Toolkit 2 as described above
2. Connect your device via USB or network
3. Ensure the RKNN driver is loaded (`lsmod | grep rknn`)
4. Copy model and test images to device

### CVITEK CV1800B
1. Setup TPU-MLIR environment using Docker
2. Source environment variables:
   ```bash
   source ./tpu-mlir/envsetup.sh
   ```
3. Copy model and test images to device

## Models

### MobileNetV2
- Input: 224x224x3 RGB
- Preprocessing: normalize with mean=[0,0,0], scale=[0.017,0.017,0.017]
- Supported formats: ONNX

### YOLOv5s
- Input: 640x640x3 RGB
- Preprocessing: normalize with scale=[1/255,1/255,1/255]
- Supported formats: ONNX

## Contributing

Contributions welcome! Key areas:
- Additional hardware support
- Improved detection methods
- New benchmark metrics
- Documentation improvements

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
