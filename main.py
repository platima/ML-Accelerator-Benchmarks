"""
Universal ML Benchmark Suite
Main orchestrator script that handles all benchmark types
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from hardware_detect import HardwareDetector
from tpu_benchmark import TPUBenchmark
from python_benchmark import PythonBenchmark
from memory_benchmark import MemoryBenchmark

class BenchmarkOrchestrator:
    def __init__(self):
        self.hardware = HardwareDetector()
        self.available_runners = self._detect_available_runners()
        self.results = {}

    def _detect_available_runners(self) -> List[str]:
        """Detect which benchmark runners are available"""
        runners = ["memory"]  # Memory benchmark always available
        
        # Check for TPU/NPU capabilities
        if any(acc["type"] in ["NPU", "TPU"] 
               for acc in self.hardware.capabilities["accelerators"]):
            runners.append("tpu")
            
        # Check for MicroPython
        try:
            import micropython
            runners.append("micropython")
        except ImportError:
            pass

        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                runners.append("cuda")
        except ImportError:
            pass

        return runners

    def display_menu(self):
        """Display available benchmark options"""
        print("\nAvailable Benchmarks:")
        print("=" * 40)
        for i, runner in enumerate(self.available_runners, 1):
            print(f"{i}. {runner.upper()} Benchmark")
        print("0. Run All Available")
        print("q. Quit")

    def run_tpu_benchmark(self) -> Dict:
        """Run TPU/NPU benchmarks"""
        print("\nRunning TPU/NPU benchmarks...")
        
        benchmark = TPUBenchmark()
        results = {}
        
        # Standard test models
        models = [
            {
                "name": "mobilenetv2",
                "path": "models/mobilenetv2-12.onnx"
            },
            {
                "name": "yolov5s",
                "path": "models/yolov5s.onnx"
            }
        ]
        
        # Standard test images
        test_images = [
            "images/cat.jpg",
            "images/dog.jpg",
            "images/person.jpg"
        ]
        
        for model in models:
            try:
                model_results = benchmark.run_model_benchmark(
                    model_name=model["name"],
                    model_path=model["path"],
                    test_images=test_images,
                    num_runs=100,
                    warm_up=10
                )
                results[model["name"]] = model_results
            except Exception as e:
                print(f"Error benchmarking {model['name']}: {str(e)}")
                
        return results

    def run_memory_benchmark(self) -> Dict:
        """Run memory benchmarks"""
        print("\nRunning memory benchmarks...")
        benchmark = MemoryBenchmark()
        return benchmark.run_benchmark()

    def run_micropython_benchmark(self) -> Dict:
        """Run MicroPython benchmarks"""
        print("\nRunning MicroPython benchmarks...")
        # Import here to avoid errors on systems without MicroPython
        from micropython_ulab_runner import UniversalBenchmark
        benchmark = UniversalBenchmark()
        return benchmark.run()

    def run_benchmark(self, choice: str) -> Optional[Dict]:
        """Run a specific benchmark"""
        if choice == "tpu":
            return self.run_tpu_benchmark()
        elif choice == "memory":
            return self.run_memory_benchmark()
        elif choice == "micropython":
            return self.run_micropython_benchmark()
        return None

    def run_all_benchmarks(self) -> Dict:
        """Run all available benchmarks"""
        results = {}
        for runner in self.available_runners:
            result = self.run_benchmark(runner)
            if result:
                results[runner] = result
        return results

    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def generate_visualizations(self, results: Dict, output_dir: str = "benchmark_results"):
        """Generate benchmark visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # TPU/NPU performance comparison
        if "tpu" in results:
            self._plot_tpu_performance(results["tpu"], output_dir)
            
        # Memory benchmark visualization
        if "memory" in results:
            self._plot_memory_results(results["memory"], output_dir)
            
        # Combined performance overview
        self._plot_overview(results, output_dir)

    def _plot_tpu_performance(self, results: Dict, output_dir: str):
        """Plot TPU/NPU benchmark results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("TPU/NPU Performance Analysis")
        
        models = list(results.keys())
        fps = [r["performance"]["throughput_fps"] for r in results.values()]
        inference = [r["performance"]["inference_times_stats"]["mean"] 
                    for r in results.values()]
        
        # FPS Comparison
        ax1.bar(models, fps)
        ax1.set_title("Throughput")
        ax1.set_ylabel("FPS")
        plt.xticks(rotation=45)
        
        # Inference Time
        ax2.bar(models, inference)
        ax2.set_title("Inference Time")
        ax2.set_ylabel("ms")
        plt.xticks(rotation=45)
        
        # Memory Usage
        if "avg_memory_mb" in results[models[0]]["performance"]:
            memory = [r["performance"]["avg_memory_mb"] for r in results.values()]
            ax3.bar(models, memory)
            ax3.set_title("Memory Usage")
            ax3.set_ylabel("MB")
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tpu_performance.png"))
        plt.close()

    def _plot_memory_results(self, results: Dict, output_dir: str):
        """Plot memory benchmark results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Memory Benchmark Results")
        
        # Bandwidth
        ax1.plot(results["array_sizes_mb"], results["read_bandwidth"], 
                label="Read")
        ax1.plot(results["array_sizes_mb"], results["write_bandwidth"], 
                label="Write")
        ax1.set_title("Memory Bandwidth")
        ax1.set_xlabel("Array Size (MB)")
        ax1.set_ylabel("MB/s")
        ax1.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_benchmark.png"))
        plt.close()

    def _plot_overview(self, results: Dict, output_dir: str):
        """Plot overall benchmark overview"""
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle("Benchmark Overview")
        
        # Create overview metrics
        # TODO: Add more comparative metrics
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "benchmark_overview.png"))
        plt.close()

    def run(self):
        """Main benchmark loop"""
        print("\nUniversal ML Benchmark Suite")
        print("=" * 40)
        
        # Print hardware capabilities
        self.hardware.print_capabilities()

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
                    runner = self.available_runners[choice-1]
                    results = {runner: self.run_benchmark(runner)}
                else:
                    print("Invalid choice")
                    continue

                if results:
                    self.save_results(results)
                    self.generate_visualizations(results)

            except ValueError:
                print("Invalid input")
                continue

def main():
    parser = argparse.ArgumentParser(description='Universal ML Benchmark Suite')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--visualizations', type=str, default='benchmark_results',
                       help='Output directory for visualizations')
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()
