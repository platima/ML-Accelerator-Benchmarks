# Benchmarks Overview

The suite provides three benchmark scripts, one for each supported Python runtime. Each performs the same core workload — matrix multiplication and element-wise operations — adapted to the runtime's available libraries and hardware APIs.

## Benchmark Scripts

| Script | Runtime | NumPy Library | Target Hardware |
|--------|---------|---------------|-----------------|
| `python-benchmark.py` | CPython 3.9+ | NumPy | Linux SBCs, desktops |
| `micropython-benchmark.py` | MicroPython 1.19+ | ulab | RP2350, ESP32 variants |
| `circuitpython-benchmark.py` | CircuitPython 8.2+ | ulab | RP2040, RP2350, ESP32-P4 |

## Core Workload

All three scripts perform the same mathematical operations:

1. **Matrix multiplication**: `np.dot(A, A)` for each channel (default 3 channels)
2. **Element-wise scaling**: `result *= 0.5`
3. **Accumulation**: sum across channels
4. **Normalisation**: `output *= (1.0 / channels)`

The matrix size is automatically determined by probing available memory with a binary search.

## Output Format

All benchmarks produce structured JSON output with four sections:

- `_meta` — source version, test date, tester, firmware info
- `device` — board type, CPU frequency, core count, sensor availability
- `performance` — array size, memory usage, timing statistics, throughput
- `benchmark` — derived metrics: total ops, ops/second, normalised score

See [Understanding Results](results.md) for detailed explanations of each field.
