"""
MicroPython Text Visualisation

Provides lightweight ASCII bar charts and summary tables for displaying
benchmark results directly on a MicroPython or CircuitPython REPL.  No
external dependencies — works with the built-in ``print()`` function.

The data format follows the ``_meta``, ``device``, ``performance``, and
``benchmark`` structure produced by the benchmark scripts.
"""


class TextVisualiser:
    """ASCII-art visualiser for benchmark results.

    Args:
        width: Maximum character width for bar charts.
    """

    def __init__(self, width=40):
        self.width = width

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_number(n):
        """Format a number with K / M / G suffix.

        Args:
            n: Numeric value.

        Returns:
            A short human-readable string.
        """
        if abs(n) >= 1e9:
            return "{:.1f}G".format(n / 1e9)
        elif abs(n) >= 1e6:
            return "{:.1f}M".format(n / 1e6)
        elif abs(n) >= 1e3:
            return "{:.1f}K".format(n / 1e3)
        else:
            return "{:.1f}".format(n)

    def bar(self, value, max_value):
        """Return an ASCII bar proportional to *value / max_value*.

        Args:
            value: Current value.
            max_value: The ceiling value (full bar).

        Returns:
            A string like ``[######----]``.
        """
        if max_value <= 0:
            return "[" + "-" * self.width + "]"
        fill = int((value / max_value) * self.width)
        fill = max(0, min(fill, self.width))
        return "[" + "#" * fill + "-" * (self.width - fill) + "]"

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------

    def show_result(self, result):
        """Print a single benchmark result dictionary.

        Args:
            result: A dict with ``_meta``, ``device``, ``performance``,
                and ``benchmark`` keys.
        """
        meta = result.get("_meta", {})
        device = result.get("device", {})
        perf = result.get("performance", {})
        bench = result.get("benchmark", {})

        board = device.get("board_type", "unknown")
        date = meta.get("Test date", "")

        print("")
        print("Benchmark Result — {} ({})".format(board, date))
        print("=" * self.width)

        print("  Board:      {}".format(board))
        print("  Cores:      {}".format(device.get("num_cores", "?")))
        print("  Freq:       {} MHz".format(device.get("cpu_freq_mhz", "?")))
        print("  Array:      {}".format(perf.get("array_size", "?")))
        print("  Avg time:   {} ms".format(perf.get("avg_inference_ms", "?")))
        print("  Ops/sec:    {}".format(self.fmt_number(bench.get("ops_per_second", 0))))
        print("  Score:      {:.1f}".format(bench.get("normalized_score", 0)))

        temp = perf.get("avg_temperature")
        if temp is not None:
            print("  Avg temp:   {:.1f} C".format(temp))

    def show_comparison(self, results):
        """Print a side-by-side ASCII comparison of multiple results.

        Each result is shown as a labelled bar proportional to its
        normalised score.

        Args:
            results: A list of result dictionaries.
        """
        if not results:
            print("No results to compare.")
            return

        # Find the maximum score for bar scaling
        max_score = 0
        entries = []
        for r in results:
            bench = r.get("benchmark", {})
            device = r.get("device", {})
            score = bench.get("normalized_score", 0)
            label = device.get("board_type", "unknown")
            entries.append((label, score))
            if score > max_score:
                max_score = score

        print("")
        print("Score Comparison")
        print("=" * (self.width + 20))

        for label, score in entries:
            padded = label.ljust(14)[:14]
            print("{} {} {:.0f}".format(padded, self.bar(score, max_score), score))

    def show_summary_table(self, results):
        """Print a compact text table of key metrics.

        Args:
            results: A list of result dictionaries.
        """
        if not results:
            print("No results.")
            return

        print("")
        print("{:<14} {:>8} {:>10} {:>10}".format(
            "Board", "Avg ms", "Ops/s", "Score"
        ))
        print("-" * 46)

        for r in results:
            device = r.get("device", {})
            perf = r.get("performance", {})
            bench = r.get("benchmark", {})
            print("{:<14} {:>8.1f} {:>10} {:>10.1f}".format(
                device.get("board_type", "?")[:14],
                perf.get("avg_inference_ms", 0),
                self.fmt_number(bench.get("ops_per_second", 0)),
                bench.get("normalized_score", 0),
            ))
