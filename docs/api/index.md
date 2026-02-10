# API Reference

The `utils/` package provides analysis, visualisation, and results-handling utilities for working with benchmark output on CPython systems.

!!! note
    These utilities require CPython with NumPy and matplotlib. They are **not** intended to run on MicroPython or CircuitPython devices.

## Modules

| Module | Class | Purpose |
|--------|-------|---------|
| [`benchmark_analyzer`](benchmark_analyzer.md) | `BenchmarkAnalyzer` | Load results, compute scores, generate reports |
| [`hardware_detect`](hardware_detect.md) | `HardwareDetector` | Detect ML acceleration hardware capabilities |
| [`results_handler`](results_handler.md) | `ResultsHandler` | Save, load, compare, and manage result files |
| [`visualization`](visualization.md) | `BenchmarkVisualizer` | Generate matplotlib comparison charts |

## Quick Start

```python
from utils import BenchmarkAnalyzer, ResultsHandler, BenchmarkVisualizer

# Load and analyse results
analyzer = BenchmarkAnalyzer(results_dir="results")
report = analyzer.generate_report()

# Visualise comparisons
viz = BenchmarkVisualizer(output_dir="results")
viz.create_comparison_plot(results)
```
