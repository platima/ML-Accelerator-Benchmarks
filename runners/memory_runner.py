"""
Memory Runner
Handles low-level memory operations for benchmarking
"""
from typing import Dict, Optional, Tuple
import numpy as np
import time
import gc

from .base import BenchmarkRunner, RunnerConfig, RunnerResult

class MemoryRunner(BenchmarkRunner):
    """Runner for memory bandwidth tests"""
    
    def __init__(self, config: Optional[RunnerConfig] = None):
        if config is None:
            config = RunnerConfig(
                input_shape=(1024, 1024),  # Default test size
                batch_size=1,
                num_warmup=2,
                num_runs=10
            )
        super().__init__(config)
        self.arrays = None

    def initialize(self) -> bool:
        """Initialize test arrays"""
        try:
            total_elements = np.prod(self.config.input_shape)
            self.arrays = (
                np.random.random(total_elements).astype(np.float32),
                np.random.random(total_elements).astype(np.float32)
            )
            return True
        except Exception as e:
            print(f"Failed to initialize memory arrays: {e}")
            return False

    def run_single(self, input_data: np.ndarray) -> RunnerResult:
        """Run a single memory test iteration"""
        pre_start = time.perf_counter()
        arr1, arr2 = self.arrays
        gc.collect()  # Ensure clean state
        pre_time = time.perf_counter() - pre_start

        # Measure read bandwidth
        infer_start = time.perf_counter()
        _ = arr1.copy()
        read_time = time.perf_counter() - infer_start

        # Measure write bandwidth
        write_start = time.perf_counter()
        arr1[:] = 1.0
        write_time = time.perf_counter() - write_start

        # Measure copy bandwidth
        post_start = time.perf_counter()
        arr2[:] = arr1
        copy_time = time.perf_counter() - post_start

        # Calculate bandwidths
        size_bytes = arr1.nbytes
        extra_metrics = {
            "read_bandwidth_mbs": size_bytes / (read_time * 1024 * 1024),
            "write_bandwidth_mbs": size_bytes / (write_time * 1024 * 1024),
            "copy_bandwidth_mbs": size_bytes / (copy_time * 1024 * 1024),
            "test_size_mb": size_bytes / (1024 * 1024)
        }

        return RunnerResult(
            inference_time_ms=read_time * 1000,
            pre_process_time_ms=pre_time * 1000,
            post_process_time_ms=(write_time + copy_time) * 1000,
            total_time_ms=(read_time + write_time + copy_time) * 1000,
            memory_usage_kb=size_bytes // 1024,
            extra_metrics=extra_metrics
        )

    def cleanup(self):
        """Clean up resources"""
        if self.arrays:
            self.arrays = None
            gc.collect()

def main():
    """Example usage"""
    config = RunnerConfig(
        input_shape=(2048, 2048),
        num_warmup=2,
        num_runs=10
    )
    
    with MemoryRunner(config) as runner:
        results = runner.run_benchmark(None)  # No input needed
        print("\nMemory Test Results:")
        print(f"Average Read Bandwidth: {np.mean([r['read_bandwidth_mbs'] for r in results['extra_metrics']]):.2f} MB/s")
        print(f"Average Write Bandwidth: {np.mean([r['write_bandwidth_mbs'] for r in results['extra_metrics']]):.2f} MB/s")
        print(f"Average Copy Bandwidth: {np.mean([r['copy_bandwidth_mbs'] for r in results['extra_metrics']]):.2f} MB/s")

if __name__ == "__main__":
    main()
