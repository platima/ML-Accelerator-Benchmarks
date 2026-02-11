"""
Hardware Detection

Detects the current platform, CPU architecture, and any ML-relevant
acceleration capabilities.  Designed to run on CPython (Linux / Windows /
macOS) as well as MicroPython and CircuitPython, where a reduced
feature set is returned.

This module is **informational only** — the benchmark scripts already
contain their own board-detection logic.  ``HardwareDetector`` is
useful for pre-flight checks and generating system-info reports.
"""

import gc
import os
import sys

try:
    import platform as _platform

    MICROPYTHON = False
except ImportError:
    _platform = None
    MICROPYTHON = True

try:
    import machine as _machine  # type: ignore[import-not-found]
except ImportError:
    _machine = None


class HardwareDetector:
    """Detect platform, CPU, memory, and optional accelerators.

    Attributes:
        platform_info: Basic platform details (OS, arch, Python impl).
        cpu_info: CPU model, frequency, core count, ISA features.
        memory_mb: Approximate total RAM in megabytes.
        accelerators: List of detected accelerator dictionaries.
    """

    def __init__(self):
        self.platform_info = self._detect_platform()
        self.cpu_info = self._detect_cpu()
        self.memory_mb = self._detect_memory()
        self.accelerators = self._detect_accelerators()

    # ------------------------------------------------------------------
    # Platform
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_platform() -> dict:
        """Return OS, architecture, and Python implementation."""
        info = {
            "os": "unknown",
            "architecture": "unknown",
            "python_impl": "unknown",
        }

        if MICROPYTHON:
            info["python_impl"] = "MicroPython"
            try:
                uname = os.uname()
                info["os"] = uname.sysname
                info["architecture"] = uname.machine
            except (AttributeError, OSError):
                pass

            # CircuitPython exposes board.board_id
            try:
                import board  # type: ignore[import-not-found]
                info["board_id"] = board.board_id
                info["python_impl"] = "CircuitPython"
            except (ImportError, AttributeError):
                pass

            # MicroPython sys.implementation._machine
            try:
                info["machine_desc"] = sys.implementation._machine
            except AttributeError:
                pass
        else:
            info["os"] = _platform.system()
            info["architecture"] = _platform.machine()
            info["python_impl"] = _platform.python_implementation()

        return info

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    def _detect_cpu(self) -> dict:
        """Return CPU model, frequency (MHz), core count, and ISA flags."""
        cpu = {
            "model": "unknown",
            "freq_mhz": 0,
            "cores": 1,
            "isa_features": [],
        }

        if MICROPYTHON:
            self._detect_cpu_micropython(cpu)
        else:
            self._detect_cpu_cpython(cpu)

        return cpu

    @staticmethod
    def _detect_cpu_micropython(cpu: dict):
        """Fill *cpu* dict with MicroPython-specific values."""
        try:
            desc = getattr(sys.implementation, "_machine", "")
            if desc:
                cpu["model"] = desc
        except AttributeError:
            pass

        try:
            uname = os.uname()
            if cpu["model"] == "unknown":
                cpu["model"] = uname.machine
        except (AttributeError, OSError):
            pass

        if _machine is not None:
            try:
                cpu["freq_mhz"] = _machine.freq() / 1_000_000
            except (AttributeError, TypeError):
                pass

    @staticmethod
    def _detect_cpu_cpython(cpu: dict):
        """Fill *cpu* dict with CPython platform values."""
        # Model
        try:
            if sys.platform == "linux":
                with open("/proc/cpuinfo", "r") as fh:
                    for line in fh:
                        if line.startswith("model name"):
                            cpu["model"] = line.split(":", 1)[1].strip()
                            break
            else:
                cpu["model"] = _platform.processor() or "unknown"
        except OSError:
            cpu["model"] = _platform.processor() or "unknown"

        # Core count
        try:
            cpu["cores"] = os.cpu_count() or 1
        except AttributeError:
            cpu["cores"] = 1

        # ISA features (Linux only — /proc/cpuinfo flags)
        try:
            if sys.platform == "linux":
                with open("/proc/cpuinfo", "r") as fh:
                    for line in fh:
                        if line.startswith("flags") or line.startswith("Features"):
                            tokens = line.split(":", 1)[1].strip().split()
                            for feat in ("neon", "sse", "sse2", "avx", "avx2", "avx512f", "rvv"):
                                if feat in tokens:
                                    cpu["isa_features"].append(feat.upper())
                            break
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_memory() -> int:
        """Return total RAM in megabytes (approximate)."""
        if MICROPYTHON:
            try:
                gc.collect()
                total = gc.mem_free() + gc.mem_alloc()
                return total // (1024 * 1024) if total > 1024 * 1024 else 0
            except AttributeError:
                return 0

        # CPython — try /proc/meminfo first, then psutil
        try:
            if sys.platform == "linux":
                with open("/proc/meminfo", "r") as fh:
                    for line in fh:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return kb // 1024
        except OSError:
            pass

        try:
            import psutil  # type: ignore[import-untyped]
            return int(psutil.virtual_memory().total / (1024 * 1024))
        except ImportError:
            return 0

    # ------------------------------------------------------------------
    # Accelerators
    # ------------------------------------------------------------------

    def _detect_accelerators(self) -> list:
        """Return a list of detected ML-relevant accelerator dicts.

        Each dict has at minimum ``type`` and ``model`` keys.
        """
        accs = []

        if MICROPYTHON:
            # On MCUs the "accelerator" is really just the CPU + ulab
            try:
                import ulab  # type: ignore[import-not-found]
                accs.append({
                    "type": "CPU+ulab",
                    "model": self.cpu_info.get("model", "MCU"),
                    "notes": "ulab {}".format(getattr(ulab, "__version__", "?")),
                })
            except ImportError:
                accs.append({
                    "type": "CPU",
                    "model": self.cpu_info.get("model", "MCU"),
                })
            return accs

        # --- CPython accelerator discovery ---
        accs.append({
            "type": "CPU",
            "model": self.cpu_info.get("model", "unknown"),
        })

        # Coral Edge TPU
        try:
            import importlib
            if importlib.util.find_spec("pycoral"):
                accs.append({"type": "TPU", "model": "Google Edge TPU"})
        except (ImportError, AttributeError):
            pass

        # Rockchip RKNN NPU
        if os.path.exists("/dev/rknpu") or os.path.exists("/usr/lib/librknnrt.so"):
            accs.append({"type": "NPU", "model": "Rockchip RKNN"})

        # Mali GPU / NPU
        if os.path.exists("/dev/mali0"):
            accs.append({"type": "GPU/NPU", "model": "Mali"})

        return accs

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line summary string."""
        lines = [
            "Hardware Detection Summary",
            "=" * 40,
            "OS:           {}".format(self.platform_info.get("os", "?")),
            "Architecture: {}".format(self.platform_info.get("architecture", "?")),
            "Python:       {}".format(self.platform_info.get("python_impl", "?")),
            "CPU model:    {}".format(self.cpu_info.get("model", "?")),
            "CPU cores:    {}".format(self.cpu_info.get("cores", "?")),
            "CPU freq:     {} MHz".format(self.cpu_info.get("freq_mhz", "?")),
            "RAM:          {} MB".format(self.memory_mb),
        ]

        isa = self.cpu_info.get("isa_features", [])
        if isa:
            lines.append("ISA features: {}".format(", ".join(isa)))

        if self.accelerators:
            lines.append("")
            lines.append("Accelerators:")
            for acc in self.accelerators:
                lines.append("  - {} ({})".format(acc.get("type", "?"), acc.get("model", "?")))

        return "\n".join(lines)

    def print_summary(self):
        """Print the hardware summary to stdout."""
        print(self.summary())


def main():
    """CLI entry point."""
    detector = HardwareDetector()
    detector.print_summary()


if __name__ == "__main__":
    main()
