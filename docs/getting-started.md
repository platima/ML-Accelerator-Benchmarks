# Getting Started

## Prerequisites

You'll need one of the following runtimes depending on your target hardware:

| Runtime | Target Hardware | NumPy Library |
|---------|----------------|---------------|
| CPython 3.9+ | Linux SBCs, desktops | NumPy |
| MicroPython 1.19+ | RP2350, ESP32 variants | ulab |
| CircuitPython 8.2+ | RP2040, RP2350, ESP32-P4 | ulab |

## Installation

### Clone the Repository

```bash
git clone https://github.com/platima/ML-Accelerator-Benchmarks.git
cd ML-Accelerator-Benchmarks
```

### CPython (Linux SBCs / Desktops)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python benchmarks/python-benchmark.py
```

### MicroPython

You'll need a MicroPython firmware build that includes **ulab** (a NumPy-like library for MicroPython). [Pimoroni firmware](https://github.com/pimoroni/pimoroni-pico-rp2350/releases) for RP2350 includes ulab by default.

1. Flash the appropriate firmware to your board
2. Copy `benchmarks/micropython-benchmark.py` to your device
3. Run via the REPL or your preferred IDE (e.g. Thonny)

### CircuitPython

CircuitPython 8.2+ includes ulab. Download the appropriate firmware from [circuitpython.org](https://circuitpython.org/).

1. Flash CircuitPython firmware to your board
2. Copy `benchmarks/circuitpython-benchmark.py` to the device as `code.py` (or run manually)
3. The benchmark will execute automatically on boot, or run via the REPL

## What Happens When You Run a Benchmark

1. **Hardware detection** — the script identifies your board type, CPU frequency, core count, and available sensors
2. **Memory probing** — a binary search finds the largest matrix size your device can handle
3. **Warmup** — a few throwaway runs to warm up caches and JIT (if applicable)
4. **Benchmark** — 10 timed iterations of matrix multiply + element-wise operations
5. **Results** — JSON output printed to stdout with device info, performance metrics, and normalised scores

## Next Steps

- [Running Benchmarks](benchmarks/running.md) — detailed usage for each runtime
- [Understanding Results](benchmarks/results.md) — what each metric means
- [Hardware Compatibility](benchmarks/hardware.md) — tested platforms and detection methods
