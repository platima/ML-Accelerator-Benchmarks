import numpy as np
import time
import psutil
import gc
from pathlib import Path

class MatrixBenchmark:
    def __init__(self):
        self.results = {}
        self.temp_path = "/sys/class/thermal/thermal_zone0/temp"
        
    def get_temperature(self):
        """Get CPU temperature on Linux systems"""
        try:
            with open(self.temp_path, 'r') as f:
                return float(f.read().strip()) / 1000.0
        except:
            return 0.0

    def get_memory_usage(self):
        """Get current memory usage"""
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def _try_create_array(self, size):
        """Try to create arrays of given size"""
        try:
            gc.collect()
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            del a, b
            return True
        except:
            return False

    def _find_max_dimensions(self):
        """Binary search for maximum supported dimensions"""
        print("\nFinding maximum supported dimensions...")
        print("--------------------------------------")
        
        size = 16
        last_success = None
        
        print("Phase 1: Finding upper bound")
        while True:
            print(f"Testing {size}x{size}...", end=" ")
            if self._try_create_array(size):
                print("OK")
                last_success = size
                size *= 2
            else:
                print("FAILED")
                break

        print("\nPhase 2: Binary search for exact maximum")
        low = last_success if last_success else size // 2
        high = size
        
        while low < high - 1:
            mid = (low + high) // 2
            print(f"Testing {mid}x{mid}...", end=" ")
            if self._try_create_array(mid):
                print("OK")
                low = mid
            else:
                print("FAILED")
                high = mid
                
        return low

    def run_benchmark(self, size=None):
        """Run matrix operations benchmark"""
        if size is None:
            size = self._find_max_dimensions()
            
        print(f"\nRunning benchmark with {size}x{size} matrices")
        print("----------------------------------------")
        
        # Create test matrices
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        # Matrix multiplication
        start_temp = self.get_temperature()
        start_mem = self.get_memory_usage()
        start_time = time.time()
        
        c = np.matmul(a, b)
        
        end_time = time.time()
        end_mem = self.get_memory_usage()
        end_temp = self.get_temperature()
        
        self.results['matmul'] = {
            'time': end_time - start_time,
            'memory_delta': end_mem - start_mem,
            'temp_delta': end_temp - start_temp
        }
        
        # Element-wise operations
        start_temp = self.get_temperature()
        start_mem = self.get_memory_usage()
        start_time = time.time()
        
        d = a + b
        e = a * b
        f = np.sqrt(a)
        
        end_time = time.time()
        end_mem = self.get_memory_usage()
        end_temp = self.get_temperature()
        
        self.results['element_ops'] = {
            'time': end_time - start_time,
            'memory_delta': end_mem - start_mem,
            'temp_delta': end_temp - start_temp
        }
        
        self._print_results()
        
    def _print_results(self):
        """Print benchmark results"""
        print("\nBenchmark Results")
        print("----------------")
        for name, metrics in self.results.items():
            print(f"\n{name}:")
            print(f"  Time: {metrics['time']:.4f} seconds")
            print(f"  Memory Delta: {metrics['memory_delta']:.2f} MB")
            print(f"  Temperature Delta: {metrics['temp_delta']:.2f}°C")

if __name__ == "__main__":
    benchmark = MatrixBenchmark()
    benchmark.run_benchmark()