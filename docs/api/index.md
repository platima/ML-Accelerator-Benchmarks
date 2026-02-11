# API Reference

The `utils/` package provides analysis, visualisation, and results-handling utilities for working with benchmark output.

!!! note
    The `visualization` module requires CPython with matplotlib.  All other modules work on MicroPython and CircuitPython as well.

## Modules

| Module | Key Class | Purpose |
|--------|-----------|---------|
| [`benchmark_analyzer`](benchmark_analyzer.md) | `BenchmarkAnalyser` | Load results, compute scores, generate reports |
| [`hardware_detect`](hardware_detect.md) | `HardwareDetector` | Detect platform, CPU, memory, and accelerators |
| [`results_handler`](results_handler.md) | `ResultsHandler` | Save, load, compare, and manage result files |
| [`visualization`](visualization.md) | `BenchmarkVisualiser` | Generate matplotlib comparison charts (CPython only) |
| [`micropython_viz`](micropython_viz.md) | `TextVisualiser` | ASCII bar charts for MicroPython / CircuitPython REPL |

## Quick Start

```python
from utils import BenchmarkAnalyser, ResultsHandler, BenchmarkVisualiser

# Load and analyse results
analyser = BenchmarkAnalyser(results_dir="results")
report = analyser.generate_report()

# Visualise comparisons (requires matplotlib)
vis = BenchmarkVisualiser(output_dir="results")
vis.create_all_charts(analyser.results)
```
