"""
MicroPython Benchmark Implementation
Tests matrix operations performance on MicroPython platforms
"""
from typing import Dict, Optional
import gc
import numpy as np

from .base import BenchmarkBase, BenchmarkConfig
from runners import MicroPythonRunner, RunnerConfig

class MicroPythonBenchmark(BenchmarkBase):
    """Benchmark for MicroPython matrix operations"""

    # Standard matrix operation configurations for different MCUs
    MCU_CONFIGS = {
        "rp2040": {
            "input_shape": (64, 64),  # Conservative for RP2040
            "num_warmup": 2,
            "num_runs": 10
        },
        "rp2350": {
            "input_shape": (128, 128),  # More capable
            "num_warmup": 2,
            "num_runs": 10
        },
        "default": {
            "input_shape": (32, 32),  # Very conservative default
            "num_warmup": 2,
            "num_runs": 5
        }
    }

    def __init__(self, 
                 mcu_type: str = "default",
                 config: Optional[BenchmarkConfig] = None):
        if config is None:
            # Use MCU-specific configuration
            mcu_config = self.MCU_CONFIGS.get(mcu_type, self.MCU_CONFIGS["default"])
            runner_config = RunnerConfig(
                input_shape=mcu_config["input_shape"],
                num_warmup=mcu_config["num_warmup"],
                num_runs=mcu_config["num_runs"]
            )
            config = BenchmarkConfig(
                name=f"MicroPython_{mcu_type}",
                runner_config=runner_config
            )
        super().__init__(config)
        
        self.mcu_type = mcu_type
        self.runner = None
        self.initial_mem = None

    def initialize(self) -> bool:
        """Initialize MicroPython runner and resources"""
        try:
            # Record initial memory state
            gc.collect()
            self.initial_mem = gc.mem_free()
            
            # Initialize runner
            self.runner = MicroPythonRunner(self.config.runner_config)
            return True
        except Exception as e:
            print(f"Failed to initialize MicroPython benchmark: {e}")
            return False

    def run(self) -> Dict:
        """Run complete benchmark suite"""
        if not self.runner:
            raise RuntimeError("MicroPython runner not initialized")
            
        results = {
            "mcu_type": self.mcu_type,
            "matrix_size": self.config.runner_config.input_shape,
            "timings": {
                "inference_ms": [],
                "pre_process_ms": [],
                "post_process_ms": [],
                "total_ms": []
            },
            "memory": {
                "initial": self.initial_mem,
                "usage": []
            },
            "hardware_info": self._get_hardware_info()
        }
        
        # Run benchmark
        run_results = self.runner.run_benchmark(None)  # No input needed
        
        # Process results
        results["timings"] = run_results["timings"]
        results["timings_stats"] = run_results["timings_stats"]
        
        if "memory" in run_results:
            results["memory"]["usage"] = run_results["memory"]
            results["memory"]["avg_usage"] = float(np.mean(run_results["memory"]))
            
        if "extra_metrics" in run_results:
            results["hardware_info"].update({
                metric: values[0] for metric, values in 
                run_results["extra_metrics"].items()
            })
            
        # Calculate throughput
        avg_total = np.mean(results["timings"]["total_ms"])
        results["throughput_ops"] = 1000.0 / avg_total  # Operations per second
        
        return results

    def _get_hardware_info(self) -> Dict:
        """Get MCU hardware information"""
        info = {
            "platform": "MicroPython",
            "mcu_type": self.mcu_type,
        }
        
        try:
            import machine
            info["cpu_freq_mhz"] = machine.freq() // 1_000_000
        except:
            pass
            
        return info

    def print_results(self, results: Dict):
        """Print MicroPython benchmark results"""
        print(f"\nMicroPython Benchmark Results ({self.mcu_type})")
        print("=" * 40)
        print(f"Matrix Size: {results['matrix_size']}")
        if "cpu_freq_mhz" in results["hardware_info"]:
            print(f"CPU Frequency: {results['hardware_info']['cpu_freq_mhz']} MHz")
            
        print("\nPerformance:")
        print(f"Operations/second: {results['throughput_ops']:.2f}")
        print(f"Average inference time: {results['timings_stats']['inference_ms']['mean']:.2f} ms")
        
        if "memory" in results:
            print("\nMemory:")
            print(f"Initial: {results['memory']['initial']} bytes")
            if "avg_usage" in results["memory"]:
                print(f"Average Usage: {results['memory']['avg_usage']} bytes")

    def cleanup(self):
        """Clean up resources"""
        if self.runner:
            self.runner.cleanup()
            self.runner = None
        gc.collect()

def main():
    """Example usage"""
    # Test different MCU configurations
    mcu_types = ["rp2040", "rp2350", "default"]
    
    for mcu_type in mcu_types:
        print(f"\nTesting {mcu_type}...")
        benchmark = MicroPythonBenchmark(mcu_type=mcu_type)
        
        try:
            with benchmark:
                results = benchmark.run()
                benchmark.print_results(results)
                if benchmark.config.save_results:
                    benchmark.save_results(results)
        except Exception as e:
            print(f"Error benchmarking {mcu_type}: {e}")

if __name__ == "__main__":
    main()
