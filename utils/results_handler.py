"""
Results Handler

Save, load, list, and compare benchmark result JSON files.  Works on
both CPython and MicroPython (the ``json`` module is available in both).

The result format uses the ``_meta``, ``device``, ``performance`` and
``benchmark`` sections produced by the benchmark scripts.
"""

import gc
import json
import os
from time import localtime

try:
    from datetime import datetime

    MICROPYTHON = False
except ImportError:
    MICROPYTHON = True


class ResultsHandler:
    """Manage benchmark result files in a directory.

    Args:
        results_dir: Directory where result JSON files are stored.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self._ensure_dir_exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir_exists(self):
        """Create the results directory if it does not exist."""
        try:
            os.mkdir(self.results_dir)
        except OSError:
            pass

    @staticmethod
    def _get_timestamp() -> str:
        """Return a ``YYYY-MM-DD`` date string."""
        if MICROPYTHON:
            t = localtime()
            return "{}-{:02d}-{:02d}".format(t[0], t[1], t[2])
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _sanitise_filename(name: str) -> str:
        """Replace characters that are unsafe in filenames."""
        unsafe = '<>:"/\\|?*'
        for ch in unsafe:
            name = name.replace(ch, "_")
        return name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_result(self, result: dict, device_name: str, description: str = "") -> str:
        """Write a benchmark result dictionary to a JSON file.

        The filename follows the convention used by the benchmark scripts:
        ``YYYY-MM-DD <device> <description>.json``

        Args:
            result: The full result dictionary (must contain ``_meta``,
                ``device``, ``performance``, ``benchmark``).
            device_name: Human-readable device name.
            description: Optional extra label for the filename.

        Returns:
            The file path of the saved file, or an empty string on error.
        """
        timestamp = self._get_timestamp()
        safe_name = self._sanitise_filename(device_name)
        filename = "{} {}".format(timestamp, safe_name)
        if description:
            filename += " {}".format(self._sanitise_filename(description))
        filename += ".json"

        filepath = os.path.join(self.results_dir, filename)
        try:
            with open(filepath, "w") as fh:
                json.dump(result, fh, indent=4)
            return filepath
        except Exception as exc:
            print("Error saving results: {}".format(exc))
            return ""
        finally:
            gc.collect()

    def load_result(self, filename: str) -> dict:
        """Load a single result file by name.

        Args:
            filename: Name of the file (relative to *results_dir*).

        Returns:
            Parsed JSON dictionary, or an empty dict on error.
        """
        filepath = os.path.join(self.results_dir, filename)
        try:
            with open(filepath, "r") as fh:
                return json.load(fh)
        except Exception as exc:
            print("Error loading {}: {}".format(filename, exc))
            return {}
        finally:
            gc.collect()

    def list_results(self, filter_str: str = None) -> list:
        """List result filenames, optionally filtered by substring.

        Args:
            filter_str: If given, only filenames containing this
                substring are returned.

        Returns:
            A sorted list of matching filenames.
        """
        filenames = []
        try:
            for name in os.listdir(self.results_dir):
                if not name.endswith(".json"):
                    continue
                if name == "results-schema.json":
                    continue
                if filter_str is None or filter_str in name:
                    filenames.append(name)
        except Exception as exc:
            print("Error listing results: {}".format(exc))
        return sorted(filenames)

    def get_latest_result(self, device_name: str = None) -> dict:
        """Return the most recently dated result.

        Because filenames start with ``YYYY-MM-DD``, sorting
        alphabetically puts the newest file last.

        Args:
            device_name: Optional filter — only consider results whose
                filename contains this string.

        Returns:
            Parsed JSON dictionary, or an empty dict if nothing found.
        """
        results = self.list_results(device_name)
        if not results:
            return {}
        return self.load_result(results[-1])

    def compare_results(self, filenames: list) -> list:
        """Load several result files and return them as a list of dicts.

        Each entry includes a ``_filename`` key for identification.

        Args:
            filenames: List of filenames (relative to *results_dir*).

        Returns:
            A list of parsed result dictionaries.
        """
        loaded = []
        for name in filenames:
            data = self.load_result(name)
            if data:
                data["_filename"] = name
                loaded.append(data)
        return loaded


def main():
    """CLI demonstration — list and print the latest result."""
    handler = ResultsHandler()

    available = handler.list_results()
    if not available:
        print("No result files found in results/")
        return

    print("Available results ({}):\n".format(len(available)))
    for name in available:
        print("  {}".format(name))

    latest = handler.load_result(available[-1])
    if latest:
        print("\nLatest result ({}):\n".format(available[-1]))
        print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
