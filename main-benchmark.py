"""
Universal ML Benchmark Orchestrator
Detects available hardware and runs appropriate benchmarks
Compatible with Python 3.x and MicroPython 3.0+
"""

import sys
import json
from typing import Dict, List, Optional

# Try importing hardware detection
try:
    from hardware_detect import HardwareDetector
    HAVE_HARDWARE_DETECT = True
except ImportError:
    HAVE_HARDWARE_DETECT = False

class BenchmarkOrchestrator:
    def __init__(self):
        self.available_runners = []
        self.results = {}
        self.detect_capabilities()

    def detect_capabilities(self):
        """Detect available benchmark runners"""
        # Always available
        self.available_runners.append(("Matrix Operations", "python-runner.py"))
        self.available_runners.append(("Memory Benchmark", "memory-benchmark.py"))

        # Check for MicroPython
        try:
            import micropython
            self.available_runners.append(("MicroPython ulab", "micropython-ulab-runner.py"))
        except ImportError:
            pass

        # Check for RKNN
        try:
            import rknn.api
            self.available_runners.append(("RKNN TPU", "rknn-runner.py"))
        except ImportError:
            pass

        # Check for CVITEK
        try:
            import cvi_toolkit
            self.available_runners.append(("CVITEK TPU", "cvitek-runner.py"))
        except ImportError:
            pass

        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.available_runners.append(("CUDA GPU", "python-cuda-runner.py"))
        except ImportError:
            pass

        # Check for VideoCore (Raspberry Pi)
        if self._check_videocore():
            self.available_runners.append(("VideoCore", "videocore-runner.py"))

        # Check for NEON SIMD
        if self._check_neon():
            self.available_runners.append(("NEON SIMD", "neon-simd-runner.py"))

    def _check_videocore(self) -> bool:
        """Check if VideoCore is available"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                return any("BCM2711" in line for line in f)
        except:
            return False

    def _check_neon(self) -> bool:
        """Check if NEON SIMD is available"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                return any("neon" in line.lower() for line in f)
        except:
            return False

    def display_menu(self):
        """Display available benchmark options"""
        print("\nAvailable Benchmarks:")
        print("=" * 40)
        for i, (name, _) in enumerate(self.available_runners, 1):
            print(f"{i}. {name}")
        print("0. Run All Available")
        print("q. Quit")

    def run_benchmark(self, runner_script: str) -> Optional[Dict]:
        """Run a specific benchmark"""
        try:
            # Import and run the benchmark
            module_name = runner_script.replace(".py", "").replace("-", "_")
            benchmark_module = __import__(module_name)
            
            if hasattr(benchmark_module, 'main'):
                return benchmark_module.main()
            return None
        except Exception as e:
            print(f"Error running {runner_script}: {str(e)}")
            return None

    def run_all_benchmarks(self):
        """Run all available benchmarks"""
        results = {}
        for name, script in self.available_runners:
            print(f"\nRunning {name} benchmark...")
            result = self.run_benchmark(script)
            if result:
                results[name] = result
        return results

    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def run(self):
        """Main benchmark loop"""
        while True:
            self.display_menu()
            choice = input("\nSelect benchmark (0-{}, q to quit): ".format(
                len(self.available_runners)))

            if choice.lower() == 'q':
                break

            try:
                choice = int(choice)
                if choice == 0:
                    results = self.run_all_benchmarks()
                elif 1 <= choice <= len(self.available_runners):
                    name, script = self.available_runners[choice-1]
                    results = {name: self.run_benchmark(script)}
                else:
                    print("Invalid choice")
                    continue

                if results:
                    self.save_results(results)

            except ValueError:
                print("Invalid input")
                continue

def main():
    print("Universal ML Benchmark Suite")
    print("=" * 40)

    # Hardware detection if available
    if HAVE_HARDWARE_DETECT:
        detector = HardwareDetector()
        detector.print_capabilities()

    # Run benchmarks
    orchestrator = BenchmarkOrchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()
