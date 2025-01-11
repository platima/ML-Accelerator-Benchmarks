"""
Universal ML Benchmark Suite - MicroPython Compatible
Main orchestrator script that handles all benchmark types
"""
import os
import sys
import json
import gc

# Conditional imports based on platform
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MICROPYTHON = False
except ImportError:
    MICROPYTHON = True

# Import benchmarks with platform checks
from benchmarks import (
    BenchmarkBase,
    MemoryBenchmark,
    MicroPythonBenchmark,
)

from utils import (
    HardwareDetector,
    ResultsHandler,
    BASIC_VISUALIZER if MICROPYTHON else BenchmarkVisualizer
)

class SimplifiedArgs:
    """Lightweight argument parser for MicroPython"""
    def __init__(self):
        self.benchmark = 'all'
        self.visualize = False
        self.compare = False
        
    @staticmethod
    def parse_args():
        args = SimplifiedArgs()
        # Simple argument parsing for MicroPython
        for i, arg in enumerate(sys.argv[1:]):
            if arg == '--benchmark' and i + 2 <= len(sys.argv):
                args.benchmark = sys.argv[i + 2]
            elif arg == '--visualize':
                args.visualize = True
            elif arg == '--compare':
                args.compare = True
        return args

class BenchmarkOrchestrator:
    def __init__(self):
        self.hardware = HardwareDetector()
        self.results_handler = ResultsHandler()
        self.visualizer = BASIC_VISUALIZER() if MICROPYTHON else BenchmarkVisualizer()
        
        # Simplified benchmark registry for MicroPython
        self.benchmarks = {
            "memory": MemoryBenchmark,
            "micropython": MicroPythonBenchmark
        }
        
        if not MICROPYTHON:
            from benchmarks import TPUBenchmark, PythonBenchmark
            self.benchmarks.update({
                "python": PythonBenchmark,
                "tpu": TPUBenchmark
            })

    def _detect_available_benchmarks(self):
        """Detect which benchmarks can run on this hardware"""
        available = ["memory"]  # Memory benchmark always available
        
        capabilities = self.hardware.capabilities
        if not MICROPYTHON and any(acc["type"] in ["NPU", "TPU"] 
                                 for acc in capabilities["accelerators"]):
            available.append("tpu")
            
        available.append("micropython" if MICROPYTHON else "python")
        return available

    def display_menu(self):
        """Display available benchmark options"""
        print("\nAvailable Benchmarks:")
        print("=" * 20)
        for i, runner in enumerate(self.available_runners, 1):
            print(f"{i}. {runner.upper()}")
        print("0. Run All")
        print("q. Quit")

    def run_benchmark(self, benchmark_type):
        """Run a specific benchmark type"""
        if benchmark_type not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_type}")
            
        benchmark_class = self.benchmarks[benchmark_type]
        benchmark = benchmark_class()
        
        gc.collect()  # MicroPython memory management
        
        results = benchmark.run()
        
        # Save results with minimal memory usage
        device_name = self.hardware.capabilities["accelerators"][0]["model"]
        self.results_handler.save_result(results, device_name)
        
        # Generate basic visualizations if not on MicroPython
        if not MICROPYTHON and hasattr(self.visualizer, 'create_report_plots'):
            self.visualizer.create_report_plots(results)
        
        return results

    def run_all_benchmarks(self):
        """Run all available benchmarks"""
        results = {}
        for runner in self.available_runners:
            gc.collect()  # MicroPython memory management
            result = self.run_benchmark(runner)
            if result:
                results[runner] = result
        return results

    def save_results(self, results, filename="benchmark_results.json"):
        """Save benchmark results to file"""
        try:
            with open(filename, "w") as f:
                self._write_json(results, f)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def _write_json(self, data, f):
        """Memory-efficient JSON writing for MicroPython"""
        if MICROPYTHON:
            # Simple JSON serialization for MicroPython
            def serialize(obj):
                if isinstance(obj, dict):
                    items = [f'"{k}": {serialize(v)}' for k, v in obj.items()]
                    return "{" + ", ".join(items) + "}"
                elif isinstance(obj, (list, tuple)):
                    items = [serialize(x) for x in obj]
                    return "[" + ", ".join(items) + "]"
                elif isinstance(obj, (int, float)):
                    return str(obj)
                else:
                    return f'"{str(obj)}"'
            
            f.write(serialize(data))
        else:
            json.dump(data, f, indent=2)

    def run(self):
        """Main benchmark loop"""
        print("\nML Benchmark Suite")
        print("=" * 20)
        
        self.hardware.print_capabilities()
        self.available_runners = self._detect_available_benchmarks()

        while True:
            self.display_menu()
            choice = input("\nSelect (0-{}, q): ".format(
                len(self.available_runners)))

            if choice.lower() == 'q':
                break

            try:
                choice = int(choice)
                if choice == 0:
                    results = self.run_all_benchmarks()
                elif 1 <= choice <= len(self.available_runners):
                    runner = self.available_runners[choice-1]
                    results = {runner: self.run_benchmark(runner)}
                else:
                    print("Invalid choice")
                    continue

                if results:
                    self.save_results(results)

            except ValueError:
                print("Invalid input")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

def main():
    if MICROPYTHON:
        args = SimplifiedArgs.parse_args()
    else:
        import argparse
        parser = argparse.ArgumentParser(description='ML Benchmark Suite')
        parser.add_argument('--benchmark', 
                          choices=['all', 'memory', 'micropython', 'python', 'tpu'],
                          default='all')
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--compare', action='store_true')
        args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator()
    
    try:
        if args.benchmark == 'all':
            available = orchestrator._detect_available_benchmarks()
            results = {}
            for benchmark_type in available:
                print(f"\nRunning {benchmark_type}...")
                results[benchmark_type] = orchestrator.run_benchmark(benchmark_type)
        else:
            results = {args.benchmark: orchestrator.run_benchmark(args.benchmark)}
            
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        sys.exit(1)
    finally:
        gc.collect()  # Final cleanup

if __name__ == "__main__":
    main()