# ML Accelerator Benchmark Suite 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmarking suite for ML accelerators on various SoCs and development boards. Compare performance, power efficiency, and resource utilization across different platforms.

## Supported Hardware

### Full NPU Support
- RV1103/RV1106 (RKNN - 0.5 TOPS)
- CV1800B (TPU-MLIR - 0.5 TOPS)
- SG2002 (TPU-MLIR)
- RK3588S (RKNN - 6 TOPS)
- NXP i.MX 93 (NPU - 2 TOPS)
- TI AM67A (TIDL)

### CPU/Other Acceleration
- RK3399 (NEON SIMD)
- BCM2711 (VideoCore VI)
- SpacemiT K1
- RP2350

## Features

- Comprehensive performance metrics
- Power consumption monitoring
- Memory usage tracking
- Temperature monitoring
- Batch processing evaluation
- Comparative analysis
- Performance scoring (0-100)
- Visualization generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-accelerator-benchmark
cd ml-accelerator-benchmark
```

2. Install required packages:
```bash
pip install numpy psutil matplotlib
```

3. Install platform-specific SDKs as needed:
- RKNN Toolkit 2 for RV1103/RV1106/RK3588S
- TPU-MLIR for CV1800B/SG2002
- TensorRT for i.MX 93
- TIDL for AM67A

## Usage

### Running Benchmarks

```bash
python benchmark_runner.py \
  --model-path path/to/your/model.onnx \
  --soc-type cv1800b \
  --input-shape 1,3,224,224 \
  --batch-size 1 \
  --output results.json
```

Supported SOC types:
- `rv1106`
- `cv1800b`
- `rk3588s`
- `imx93`
- `am67a`
- `rk3399`
- `bcm2711`
- `rp2350`

### Analyzing Results

```bash
python benchmark_analyzer.py \
  --results results1.json results2.json \
  --output-dir analysis
```

This will generate:
- A detailed comparison report
- Performance visualizations
- Efficiency metrics

## Sample Output

```
SoC Benchmark Analysis Report
================================================================================
Performance Comparison
----------------------------------------
SoC      Score  Tier                 FPS    Power(W)  Temp(°C)
RK3588S   78.3  High-End (Edge)     121.9    3.2      45.2
CV1800B   72.1  High-End (Edge)     100.0    2.0      42.1
IMX93     61.4  Mid-Range (Mobile)   65.4    1.8      38.5
BCM2711   34.2  Basic (MCU)          21.9    4.5      52.3
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- Additional SoC support
- New metrics or analysis methods
- Bug fixes
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to contributors from the various SoC communities
- Model examples from TorchVision and TensorFlow
- Performance testing guidance from MLPerf

## Featured In

- [YouTube Video: Comparing ML Accelerator Performance](https://youtube.com/...)
