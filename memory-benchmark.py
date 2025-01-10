"""
Memory Bandwidth Benchmark Module
Tests memory performance for ML workloads
"""
import time
import numpy as np
from typing import Dict, Tuple, Optional

class MemoryBenchmark:
    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        
    def _create_test_arrays(self, size_mb: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create test arrays of specified size"""
        arr_size = int((size_mb * 1024 * 1024) / 8)  # Convert MB to elements (64-bit)
        return np.random.random(arr_size), np.random.random(arr_size)
        
    def _measure_bandwidth(self, arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, float]:
        """Measure different memory bandwidth metrics"""
        size_bytes = arr1.nbytes
        results = {}
        
        # Sequential read
        start = time.perf_counter()
        _ = arr1.copy()
        elapsed = time.perf_counter() - start
        results["read_bandwidth"] = size_bytes / (elapsed * 1024 * 1024)  # MB/s
        
        # Sequential write
        start = time.perf_counter()
        arr1[:] = 1.0
        elapsed = time.perf_counter() - start
        results["write_bandwidth"] = size_bytes / (elapsed * 1024 * 1024)
        
        # Copy bandwidth
        start = time.perf_counter()
        arr2[:] = arr1
        elapsed = time.perf_counter() - start
        results["copy_bandwidth"] = size_bytes / (elapsed * 1024 * 1024)
        
        return results
        
    def run_benchmark(self) -> Dict:
        """Run complete memory bandwidth benchmark"""
        results = {
            "array_sizes_mb": [],
            "read_bandwidth": [],
            "write_bandwidth": [],
            "copy_bandwidth": []
        }
        
        # Test with increasing sizes
        test_sizes = [1, 2, 4, 8, 16, 32, 64, min(128, self.max_size_mb)]
        
        for size in test_sizes:
            print(f"Testing with array size: {size}MB")
            try:
                arr1, arr2 = self._create_test_arrays(size)
                bandwidth = self._measure_bandwidth(arr1, arr2)
                
                results["array_sizes_mb"].append(size)
                results["read_bandwidth"].append(bandwidth["read_bandwidth"])
                results["write_bandwidth"].append(bandwidth["write_bandwidth"])
                results["copy_bandwidth"].append(bandwidth["copy_bandwidth"])
                
                # Clean up
                del arr1, arr2
                
            except MemoryError:
                print(f"Memory allocation failed at {size}MB")
                break
                
        # Calculate averages
        results["avg_read_bandwidth"] = np.mean(results["read_bandwidth"])
        results["avg_write_bandwidth"] = np.mean(results["write_bandwidth"])
        results["avg_copy_bandwidth"] = np.mean(results["copy_bandwidth"])
        
        return results
        
    def print_results(self, results: Dict):
        """Print benchmark results in a readable format"""
        print("\nMemory Bandwidth Test Results")
        print("=" * 40)
        print(f"Average Read Bandwidth:  {results['avg_read_bandwidth']:.2f} MB/s")
        print(f"Average Write Bandwidth: {results['avg_write_bandwidth']:.2f} MB/s")
        print(f"Average Copy Bandwidth:  {results['avg_copy_bandwidth']:.2f} MB/s")
        
        print("\nDetailed Results:")
        print("Size (MB) | Read (MB/s) | Write (MB/s) | Copy (MB/s)")
        print("-" * 55)
        for i, size in enumerate(results["array_sizes_mb"]):
            print(f"{size:9d} | {results['read_bandwidth'][i]:10.2f} | "
                  f"{results['write_bandwidth'][i]:11.2f} | "
                  f"{results['copy_bandwidth'][i]:10.2f}")

def main():
    benchmark = MemoryBenchmark()
    results = benchmark.run_benchmark()
    benchmark.print_results(results)

if __name__ == "__main__":
    main()
