"""
ML Accelerator Benchmark Utilities Package

On CPython all modules are available.  On MicroPython / CircuitPython
the matplotlib-dependent ``visualization`` module is skipped and
``micropython_viz`` is offered instead.
"""

__version__ = "0.9.0"

from .hardware_detect import HardwareDetector
from .benchmark_analyzer import BenchmarkAnalyser, BenchmarkResult, load_result
from .results_handler import ResultsHandler
from .validate_results import load_schema, validate_file, validate_directory

# micropython_viz is always safe to import (no external deps)
from .micropython_viz import TextVisualiser

# visualization requires matplotlib — guard for MicroPython
try:
    from .visualization import BenchmarkVisualiser
except ImportError:
    BenchmarkVisualiser = None  # type: ignore[misc,assignment]

__all__ = [
    "HardwareDetector",
    "BenchmarkAnalyser",
    "BenchmarkResult",
    "load_result",
    "ResultsHandler",
    "TextVisualiser",
    "BenchmarkVisualiser",
    "load_schema",
    "validate_file",
    "validate_directory",
]