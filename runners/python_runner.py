"""
Python CPU/GPU Runner
Basic matrix operations using NumPy/CuPy
"""
from typing import Dict, Optional, Union
import time
import os
import numpy as np

from .base import BenchmarkRunner, RunnerConfig, RunnerResult

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class PythonRunner(BenchmarkRunner):
    """Runner for basic Python matrix operations"""
    
    def __init__(self, config: Optional[RunnerConfig] = None, use_gpu: bool = False):
        if config is None:
            config = RunnerConfig(
                input_shape=(1024, 1024),  # Default size
                batch_size=1,
                num_warmup=5,
                num_runs=100,
                threads=os.cpu_count() or 1
            )
        super().__init__(config)
        
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.arrays = None

    def initialize(self) -> bool:
        """Initialize test arrays"""
        try:
            shape = self.config.input_shape
            
            # Create input arrays
            self.arrays = [
                self.xp.random.rand(*shape).astype(self.xp.float32)
                for _ in range(3)  # RGB channels
            ]
            
            return True
        except Exception as e:
            print(f"Failed to initialize arrays: {e}")
            return False

    def run_single(self, input_data: Union[np.ndarray, None]) -> RunnerResult:
        """Run a single iteration of matrix operations"""
        if input_data is not None and isinstance(input_data, np.ndarray):
            if self.use_gpu:
                input_data = cp.asarray(input_data)
        else:
            input_data = self.arrays[0]  # Use first test array

        pre_start = time.perf_counter()
        # Ensure data is in correct format
        if input_data.dtype != self.xp.float32:
            input_data = input_data.astype(self.xp.float32)
        pre_time = time.perf_counter() - pre_start

        infer_start = time.perf_counter()
        # Matrix operations
        result = self.xp.dot(input_data, input_data.T)
        result = self.xp.maximum(result, 0)  # ReLU
        result = self.xp.mean(result, axis=1)
        
        if self.use_gpu:
            cp.cuda.stream.get_current_stream().synchronize()
        infer_time = time.perf_counter() - infer_start

        post_start = time.perf_counter()
        # Get memory usage
        try:
            import psutil
            memory_kb = psutil.Process().memory_info().rss // 1024
            if self.use_gpu:
                memory_kb += cp.get_default_memory_pool().used_bytes() // 1024
        except:
            memory_kb = None
            
        # Convert result back to CPU if needed
        if self.use_gpu:
            result = cp.asnumpy(result)
        post_time = time.perf_counter() - post_start

        return RunnerResult(
            inference_time_ms=infer_time * 1000,
            pre_process_time_ms=pre_time * 1000,
            post_process_time_ms=post_time * 1000,
            total_time_ms=(infer_time + pre_time + post_time) * 1000,
            memory_usage_kb=memory_kb,
            output=result,
            extra_metrics={
                "device": "GPU" if self.use_gpu else "CPU",
                "threads": self.config.threads
            }
        )

    def cleanup(self):
        """Clean up resources"""
        if self.arrays:
            del self.arrays
            self.arrays = None
            
        if self.use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

def main():
    """Example usage"""
    # Test both CPU and GPU
    configs = [
        (False, "CPU"),
        (True, "GPU")
    ]
    
    for use_gpu, device in configs:
        if use_gpu and not CUDA_AVAILABLE:
            print("\nGPU testing skipped (CUDA not available)")
            continue
            
        print(f"\nTesting {device} performance:")
        config = RunnerConfig(
            input_shape=(2048, 2048),
            num_warmup=5,
            num_runs=100
        )
        
        with PythonRunner(config, use_gpu=use_gpu) as runner:
            results = runner.run_benchmark(None)
            print(f"Average Inference Time: {results['timings_stats']['inference_ms']['mean']:.2f} ms")
            print(f"Throughput: {results['throughput_fps']:.2f} FPS")
            if results.get('memory'):
                print(f"Memory Usage: {results['memory'][0] / 1024:.2f} MB")

if __name__ == "__main__":
    main()