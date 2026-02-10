#!/usr/bin/env python3
"""
Benchmark Analyser

Loads, scores, and compares benchmark result JSON files produced by the
ML Accelerator Benchmark suite.  Works with the actual result format
(``_meta``, ``device``, ``performance``, ``benchmark`` sections).

Usage::

    python -m utils.benchmark_analyzer --results results/*.json
    python -m utils.benchmark_analyzer --results-dir results/
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Parsed representation of a single benchmark result file.

    All fields map directly to the JSON structure emitted by the three
    benchmark scripts (``python-benchmark.py``, ``micropython-benchmark.py``,
    ``circuitpython-benchmark.py``).
    """

    # _meta
    source_version: str = ""
    source_code: str = ""
    source_repo: str = ""
    test_date: str = ""
    tester: str = ""
    firmware: str = ""
    notes: str = ""

    # device
    board_type: str = "unknown"
    cpu_freq_mhz: float = 0.0
    num_cores: int = 1
    temp_sensor: bool = False
    power_sensor: bool = False

    # performance
    channels: int = 3
    array_size: int = 0
    memory_total: float = 0.0
    memory_used: float = 0.0
    min_inference_ms: float = 0.0
    max_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    throughput_fps: float = 0.0
    avg_temperature: Optional[float] = None
    max_temperature: Optional[float] = None

    # benchmark
    total_ops: float = 0.0
    ops_per_second: float = 0.0
    normalized_score: float = 0.0
    theoretical_power: float = 0.0

    # convenience
    filename: str = ""


def load_result(filepath: str) -> BenchmarkResult:
    """Load a single benchmark result JSON file into a ``BenchmarkResult``.

    Args:
        filepath: Path to the ``.json`` result file.

    Returns:
        A populated ``BenchmarkResult`` instance.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(filepath, "r") as fh:
        data = json.load(fh)

    meta = data.get("_meta", {})
    device = data.get("device", {})
    perf = data.get("performance", {})
    bench = data.get("benchmark", {})

    return BenchmarkResult(
        source_version=meta.get("Source version", ""),
        source_code=meta.get("Source code", ""),
        source_repo=meta.get("Source repo", ""),
        test_date=meta.get("Test date", ""),
        tester=meta.get("Tester", ""),
        firmware=meta.get("Firmware", ""),
        notes=meta.get("Notes", ""),
        board_type=device.get("board_type", "unknown"),
        cpu_freq_mhz=device.get("cpu_freq_mhz", 0.0),
        num_cores=device.get("num_cores", 1),
        temp_sensor=device.get("temp_sensor", False),
        power_sensor=device.get("power_sensor", False),
        channels=perf.get("channels", 3),
        array_size=perf.get("array_size", 0),
        memory_total=perf.get("memory_total", 0.0),
        memory_used=perf.get("memory_used", 0.0),
        min_inference_ms=perf.get("min_inference_ms", 0.0),
        max_inference_ms=perf.get("max_inference_ms", 0.0),
        avg_inference_ms=perf.get("avg_inference_ms", 0.0),
        throughput_fps=perf.get("throughput_fps", 0.0),
        avg_temperature=perf.get("avg_temperature"),
        max_temperature=perf.get("max_temperature"),
        total_ops=bench.get("total_ops", 0.0),
        ops_per_second=bench.get("ops_per_second", 0.0),
        normalized_score=bench.get("normalized_score", 0.0),
        theoretical_power=bench.get("theoretical_power", 0.0),
        filename=os.path.basename(filepath),
    )


class BenchmarkAnalyser:
    """Load and analyse a collection of benchmark results.

    Args:
        results_dir: Directory containing ``*.json`` result files.
            If provided, all JSON files in the directory are loaded.
        result_files: Explicit list of file paths to load.
    """

    def __init__(
        self,
        results_dir: Optional[str] = None,
        result_files: Optional[List[str]] = None,
    ):
        self.results: List[BenchmarkResult] = []

        if results_dir:
            results_path = Path(results_dir)
            for fp in sorted(results_path.glob("*.json")):
                if fp.name == "results-schema.json":
                    continue
                try:
                    self.results.append(load_result(str(fp)))
                except (json.JSONDecodeError, KeyError) as exc:
                    print("Warning: skipping {}: {}".format(fp.name, exc))

        if result_files:
            for fp in result_files:
                try:
                    self.results.append(load_result(fp))
                except (json.JSONDecodeError, KeyError) as exc:
                    print("Warning: skipping {}: {}".format(fp, exc))

    def sorted_by(
        self, key: str = "normalized_score", reverse: bool = True
    ) -> List[BenchmarkResult]:
        """Return results sorted by the given attribute.

        Args:
            key: Attribute name on ``BenchmarkResult`` to sort by.
            reverse: If ``True`` (default), sort descending (best first).

        Returns:
            A new sorted list of ``BenchmarkResult`` instances.
        """
        return sorted(
            self.results, key=lambda r: getattr(r, key, 0), reverse=reverse
        )

    def generate_comparison_table(self) -> str:
        """Generate a formatted text comparison table.

        Returns:
            A multi-line string table comparing all loaded results by
            normalised score, ops/second, array size, and more.
        """
        ranked = self.sorted_by("normalized_score")

        headers = [
            "Board",
            "Date",
            "Array",
            "Avg ms",
            "Ops/s",
            "Norm Score",
            "Cores",
            "MHz",
        ]

        rows = []
        for r in ranked:
            rows.append(
                [
                    r.board_type,
                    r.test_date,
                    str(r.array_size),
                    "{:.1f}".format(r.avg_inference_ms),
                    _fmt_large(r.ops_per_second),
                    "{:.1f}".format(r.normalized_score),
                    str(r.num_cores),
                    "{:.0f}".format(r.cpu_freq_mhz),
                ]
            )

        col_widths = [
            max(len(h), *(len(row[i]) for row in rows))
            for i, h in enumerate(headers)
        ]

        lines = []
        header_line = "  ".join(
            h.ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        lines.append(header_line)
        lines.append("-" * len(header_line))

        for row in rows:
            lines.append(
                "  ".join(
                    cell.ljust(col_widths[i]) for i, cell in enumerate(row)
                )
            )

        return "\n".join(lines)

    def generate_report(self) -> str:
        """Generate a full-text analysis report.

        Returns:
            A multi-line report string with comparison table and per-device
            detail sections.
        """
        lines = [
            "ML Accelerator Benchmark Analysis Report",
            "=" * 60,
            "",
            "Comparison Table",
            "-" * 60,
            self.generate_comparison_table(),
            "",
        ]

        for r in self.sorted_by("normalized_score"):
            lines.append("")
            lines.append("Device: {} ({})".format(r.board_type, r.test_date))
            lines.append("-" * 40)
            lines.append("  Version:      {}".format(r.source_version))
            lines.append("  Firmware:     {}".format(r.firmware))
            lines.append("  Cores:        {}".format(r.num_cores))
            lines.append("  Frequency:    {:.0f} MHz".format(r.cpu_freq_mhz))
            lines.append(
                "  Array size:   {}x{}".format(r.array_size, r.array_size)
            )
            lines.append(
                "  Memory used:  {} bytes".format(int(r.memory_used))
            )
            lines.append(
                "  Avg time:     {:.3f} ms".format(r.avg_inference_ms)
            )
            lines.append(
                "  Ops/second:   {}".format(_fmt_large(r.ops_per_second))
            )
            lines.append(
                "  Norm. score:  {:.2f}".format(r.normalized_score)
            )

            if r.avg_temperature is not None:
                lines.append(
                    "  Avg temp:     {:.1f} °C".format(r.avg_temperature)
                )
            if r.notes:
                lines.append("  Notes:        {}".format(r.notes))

        return "\n".join(lines)


def _fmt_large(n: float) -> str:
    """Format a large number with K/M/G suffix."""
    if abs(n) >= 1e9:
        return "{:.2f}G".format(n / 1e9)
    elif abs(n) >= 1e6:
        return "{:.2f}M".format(n / 1e6)
    elif abs(n) >= 1e3:
        return "{:.2f}K".format(n / 1e3)
    else:
        return "{:.2f}".format(n)


def main():
    """CLI entry point for analysing benchmark results."""
    parser = argparse.ArgumentParser(
        description="Analyse ML Accelerator Benchmark results"
    )
    parser.add_argument(
        "--results", nargs="+", help="Paths to individual result JSON files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing result JSON files (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write report to this file instead of stdout",
    )
    args = parser.parse_args()

    # Default to results/ directory if nothing specified
    results_dir = args.results_dir
    if not args.results and not results_dir:
        results_dir = "results"

    analyser = BenchmarkAnalyser(
        results_dir=results_dir, result_files=args.results
    )

    if not analyser.results:
        print("No valid result files found.")
        return

    report = analyser.generate_report()

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report)
        print("Report written to {}".format(args.output))
    else:
        print(report)


if __name__ == "__main__":
    main()
