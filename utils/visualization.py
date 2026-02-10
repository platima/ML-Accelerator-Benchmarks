#!/usr/bin/env python3
"""
Visualisation Module

Generates matplotlib charts for benchmark results produced by the
ML Accelerator Benchmark suite.  Expects data loaded via
:func:`benchmark_analyzer.load_result` or the raw JSON structure
(``_meta``, ``device``, ``performance``, ``benchmark``).

Requires: matplotlib, numpy.

Usage::

    from utils.benchmark_analyzer import BenchmarkAnalyser
    from utils.visualization import BenchmarkVisualiser

    analyser = BenchmarkAnalyser(results_dir="results")
    vis = BenchmarkVisualiser()
    vis.create_comparison_chart(analyser.results)
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .benchmark_analyzer import BenchmarkResult


class BenchmarkVisualiser:
    """Create charts from a list of ``BenchmarkResult`` objects.

    Args:
        output_dir: Directory where generated images are saved.
    """

    # Colour palette suitable for colour-blind readers.
    COLOURS = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def create_comparison_chart(
        self,
        results: List[BenchmarkResult],
        output_file: str = "comparison.png",
    ) -> Path:
        """Create a four-panel comparison chart.

        Panels:
            1. Average inference time (ms)  – lower is better
            2. Throughput (FPS)              – higher is better
            3. Ops per second               – higher is better
            4. Normalised score             – higher is better

        Args:
            results: List of ``BenchmarkResult`` instances.
            output_file: Filename for the saved chart.

        Returns:
            Path to the saved image.
        """
        labels = [r.board_type for r in results]
        colours = [self.COLOURS[i % len(self.COLOURS)] for i in range(len(results))]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("ML Accelerator Benchmark Comparison", fontsize=14)

        # 1. Average inference time
        values = [r.avg_inference_ms for r in results]
        ax1.barh(labels, values, color=colours)
        ax1.set_xlabel("Avg Inference Time (ms)")
        ax1.set_title("Inference Time (lower is better)")
        ax1.invert_xaxis()

        # 2. Throughput
        values = [r.throughput_fps for r in results]
        ax2.barh(labels, values, color=colours)
        ax2.set_xlabel("Throughput (FPS)")
        ax2.set_title("Throughput (higher is better)")

        # 3. Ops per second
        values = [r.ops_per_second for r in results]
        ax3.barh(labels, values, color=colours)
        ax3.set_xlabel("Ops / Second")
        ax3.set_title("Ops per Second (higher is better)")

        # 4. Normalised score
        values = [r.normalized_score for r in results]
        ax4.barh(labels, values, color=colours)
        ax4.set_xlabel("Normalised Score")
        ax4.set_title("Normalised Score (higher is better)")

        plt.tight_layout()
        dest = self.output_dir / output_file
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        return dest

    def create_score_bar_chart(
        self,
        results: List[BenchmarkResult],
        output_file: str = "scores.png",
    ) -> Path:
        """Create a single horizontal bar chart ranked by normalised score.

        Args:
            results: List of ``BenchmarkResult`` instances.
            output_file: Filename for the saved chart.

        Returns:
            Path to the saved image.
        """
        ranked = sorted(results, key=lambda r: r.normalized_score)
        labels = [r.board_type for r in ranked]
        scores = [r.normalized_score for r in ranked]
        colours = [self.COLOURS[i % len(self.COLOURS)] for i in range(len(ranked))]

        fig, ax = plt.subplots(figsize=(10, max(4, len(ranked) * 0.6)))
        ax.barh(labels, scores, color=colours)
        ax.set_xlabel("Normalised Score")
        ax.set_title("ML Accelerator Benchmark — Normalised Score Ranking")

        for i, (score, label) in enumerate(zip(scores, labels)):
            ax.text(score + max(scores) * 0.01, i, " {:.0f}".format(score), va="center")

        plt.tight_layout()
        dest = self.output_dir / output_file
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        return dest

    def create_memory_chart(
        self,
        results: List[BenchmarkResult],
        output_file: str = "memory.png",
    ) -> Path:
        """Create a stacked bar chart showing memory total vs used.

        Args:
            results: List of ``BenchmarkResult`` instances.
            output_file: Filename for the saved chart.

        Returns:
            Path to the saved image.
        """
        labels = [r.board_type for r in results]
        used = np.array([r.memory_used for r in results])
        free = np.array([r.memory_total - r.memory_used for r in results])

        fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.6)))
        ax.barh(labels, used, label="Used", color=self.COLOURS[3])
        ax.barh(labels, free, left=used, label="Free", color=self.COLOURS[0])
        ax.set_xlabel("Memory (bytes)")
        ax.set_title("Memory Usage")
        ax.legend()

        plt.tight_layout()
        dest = self.output_dir / output_file
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        return dest

    def create_temperature_chart(
        self,
        results: List[BenchmarkResult],
        output_file: str = "temperature.png",
    ) -> Optional[Path]:
        """Create a grouped bar chart of average and max temperature.

        Only includes results that have temperature data.

        Args:
            results: List of ``BenchmarkResult`` instances.
            output_file: Filename for the saved chart.

        Returns:
            Path to the saved image, or ``None`` if no temperature data.
        """
        filtered = [r for r in results if r.avg_temperature is not None]
        if not filtered:
            return None

        labels = [r.board_type for r in filtered]
        avg_temps = [r.avg_temperature for r in filtered]
        max_temps = [r.max_temperature if r.max_temperature is not None else 0 for r in filtered]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, avg_temps, width, label="Avg °C", color=self.COLOURS[0])
        ax.bar(x + width / 2, max_temps, width, label="Max °C", color=self.COLOURS[3])
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature During Benchmark")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        dest = self.output_dir / output_file
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        return dest

    def create_all_charts(
        self,
        results: List[BenchmarkResult],
    ) -> List[Path]:
        """Generate all available charts and return their paths.

        Args:
            results: List of ``BenchmarkResult`` instances.

        Returns:
            A list of ``Path`` objects for each chart created.
        """
        paths: List[Path] = []
        paths.append(self.create_comparison_chart(results))
        paths.append(self.create_score_bar_chart(results))
        paths.append(self.create_memory_chart(results))

        temp_path = self.create_temperature_chart(results)
        if temp_path is not None:
            paths.append(temp_path)

        return paths


def main():
    """CLI entry point — generate charts from the default results directory."""
    from .benchmark_analyzer import BenchmarkAnalyser

    analyser = BenchmarkAnalyser(results_dir="results")
    if not analyser.results:
        print("No result files found in results/")
        return

    vis = BenchmarkVisualiser()
    paths = vis.create_all_charts(analyser.results)
    for p in paths:
        print("Saved: {}".format(p))


if __name__ == "__main__":
    main()
