"""
Memory Bandwidth Benchmark Implementation
Tests memory performance for ML workloads
"""
from typing import Dict, Optional
import numpy as np

from .base import BenchmarkBase, BenchmarkConfig
from runners import MemoryRunner, RunnerConfig

class MemoryBenchmark(BenchmarkBase):
    """Memory bandwidth benchmark implementation"""

    # Standard test configurations
    DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]  # MB sizes to test
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        if config is None:
            runner_config = RunnerConfig(
                input_shape=(1024, 1024),  # Default size
                num_warmup=2,
                num_runs=10
            )
            config = BenchmarkConfig(
                name="Memory",
                runner_config=runner_config,
                warmup_runs=2,
                num_runs=10
            )
        super().__init__(config)
        
        self.runner = None
        self.test_sizes = [size for size in self.DEFAULT_SIZES 
                          if size <= self.config.runner_config.input_shape[0]]

    def initialize(self) -> bool:
        """Initialize memory runner"""
        try:
            self.runner = MemoryRunner(self.config.runner_config)
            return True
        except Exception as e:
            print(f"Failed to initialize memory benchmark: {e}")
            return False

    def run(self) -> Dict:
        """Run complete memory bandwidth benchmark suite"""
        if not self.runner:
            raise RuntimeError("Memory runner not initialized")

        results = {
            "array_sizes_mb": [],
            "read_bandwidth": [],
            "write_bandwidth": [],
            "copy_bandwidth": [],
            "timings": {
                "total_ms": []
            }
        }

        # Test different array sizes
        for size_mb in self.test_sizes:
            print(f"\nTesting with array size: {size_mb}MB")
            
            try:
                # Set array size for this test
                array_elements = int(size_mb * 1024 * 1024 / 4)  # Convert MB to float32 elements
                self.runner.config.input_shape = (array_elements, 1)
                
                # Run benchmark
                run_result = self.runner.run_benchmark(None)  # No input needed
                
                # Extract results
                results["array_sizes_mb"].append(size_mb)
                results["timings"]["total_ms"].extend(run_result["timings"]["total_ms"])
                
                # Get bandwidth metrics
                bandwidths = []
                for metrics in run_result["extra_metrics"]:
                    bandwidths.append({
                        "read": metrics["read_bandwidth_mbs"],
                        "write": metrics["write_bandwidth_mbs"],
                        "copy": metrics["copy_bandwidth_mbs"]
                    })
                
                # Average bandwidths for this size
                results["read_bandwidth"].append(float(np.mean([b["read"] for b in bandwidths])))
                results["write_bandwidth"].append(float(np.mean([b["write"] for b in bandwidths])))
                results["copy_bandwidth"].append(float(np.mean([b["copy"] for b in bandwidths])))
                
            except Exception as e:
                print(f"Error testing size {size_mb}MB: {e}")
                break

        # Calculate overall statistics
        results["avg_read_bandwidth"] = float(np.mean(results["read_bandwidth"]))
        results["avg_write_bandwidth"] = float(np.mean(results["write_bandwidth"]))
        results["avg_copy_bandwidth"] = float(np.mean(results["copy_bandwidth"]))
        
        # Calculate timing statistics
        times = results["timings"]["total_ms"]
        results["timings_stats"] = {
            "total_ms": {
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "p90": float(np.percentile(times, 90))
            }
        }
        
        return results

    def print_results(self, results: Dict):
        """Print memory benchmark results"""
        print("\nMemory Bandwidth Test Results")
        print("=" * 40)
        print(f"Average Read Bandwidth:  {results['avg_read_bandwidth']:.2f} MB/s")
        print(f"Average Write Bandwidth: {results['avg_write_bandwidth']:.2f} MB/s")
        print(f"Average Copy Bandwidth:  {results['avg_copy_bandwidth']:.2f} MB/s")
        
        print("\nDetailed Results by Array Size:")
        print("Size (MB) | Read (MB/s) | Write (MB/s) | Copy (MB/s)")
        print("-" * 55)
        for i, size in enumerate(results["array_sizes_mb"]):
            print(f"{size:9d} | {results['read_bandwidth'][i]:10.2f} | "
                  f"{results['write_bandwidth'][i]:11.2f} | "
                  f"{results['copy_bandwidth'][i]:10.2f}")

    def cleanup(self):
        """Clean up resources"""
        if self.runner:
            self.runner.cleanup()
            self.runner = None

def main():
    """Example usage"""
    benchmark = MemoryBenchmark()
    
    try:
        with benchmark:
            results = benchmark.run()
            benchmark.print_results(results)
            if benchmark.config.save_results:
                benchmark.save_results(results)
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == "__main__":
    main()
