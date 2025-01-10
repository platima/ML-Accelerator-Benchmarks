#!/usr/bin/env python3
import time
import psutil
import argparse
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class BenchmarkResult:
    model_name: str
    inference_time: float  # in milliseconds
    memory_usage: float   # in MB
    power_usage: Optional[float]  # in watts, if available
    throughput: float    # inferences per second
    
class NPUBenchmark:
    def __init__(self, model_path: str, input_shape: tuple, warmup_runs: int = 5):
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("NPUBenchmark")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _measure_memory(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _measure_power(self) -> Optional[float]:
        """
        Attempt to measure power usage on supported platforms
        Returns power in watts or None if not available
        """
        try:
            # This path exists on many SBCs for power measurement
            with open('/sys/class/power_supply/BAT0/power_now', 'r') as f:
                return float(f.read()) / 1000000.0  # convert μW to W
        except:
            return None

    def run_inference(self, model, input_data: np.ndarray) -> float:
        """Run a single inference and return time in milliseconds"""
        start = time.perf_counter()
        model.predict(input_data)
        end = time.perf_counter()
        return (end - start) * 1000  # convert to ms

    def benchmark_model(self, model_name: str) -> BenchmarkResult:
        """Run complete benchmark for a single model"""
        self.logger.info(f"Starting benchmark for {model_name}")
        
        # Create dummy input data
        input_data = np.random.rand(*self.input_shape).astype(np.float32)
        
        try:
            # Load model - this would need to be adapted for specific NPU frameworks
            model = self._load_model(model_name)
            
            # Warmup runs
            self.logger.info("Performing warmup runs...")
            for _ in range(self.warmup_runs):
                self.run_inference(model, input_data)

            # Actual benchmark runs
            times = []
            memory_usage = []
            power_readings = []

            for i in range(100):  # 100 inference runs
                if i % 10 == 0:
                    self.logger.info(f"Running inference {i}/100")
                
                memory_usage.append(self._measure_memory())
                power = self._measure_power()
                if power:
                    power_readings.append(power)
                
                inference_time = self.run_inference(model, input_data)
                times.append(inference_time)

            avg_time = np.mean(times)
            avg_memory = np.mean(memory_usage)
            avg_power = np.mean(power_readings) if power_readings else None
            throughput = 1000 / avg_time  # convert ms to inferences/second

            result = BenchmarkResult(
                model_name=model_name,
                inference_time=avg_time,
                memory_usage=avg_memory,
                power_usage=avg_power,
                throughput=throughput
            )
            
            self.results.append(result)
            return result

        except Exception as e:
            self.logger.error(f"Error benchmarking {model_name}: {str(e)}")
            raise

    def _load_model(self, model_name: str):
        """
        Load model based on name. This method should be implemented
        specifically for each NPU framework (RKNN, Kendryte, etc.)
        """
        raise NotImplementedError("Model loading must be implemented for specific NPU framework")

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file"""
        results_dict = {
            "platform_info": {
                "cpu_info": self._get_cpu_info(),
                "total_memory": psutil.virtual_memory().total / (1024 * 1024),  # MB
                "os_info": self._get_os_info()
            },
            "results": [vars(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def _get_cpu_info(self) -> Dict:
        """Get CPU information"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
            return {
                "model": [line.split(': ')[1].strip() 
                         for line in cpu_info.split('\n') 
                         if 'model name' in line][0]
            }
        except:
            return {"model": "unknown"}

    def _get_os_info(self) -> Dict:
        """Get OS information"""
        import platform
        return {
            "system": platform.system(),
            "release": platform.release()
        }

def main():
    parser = argparse.ArgumentParser(description='NPU Benchmark for Resource-Constrained SBCs')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing the models to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file path for results')
    parser.add_argument('--warmup-runs', type=int, default=5,
                       help='Number of warmup runs before benchmarking')
    args = parser.parse_args()

    # Example usage - would need to be adapted for specific NPU framework
    benchmark = NPUBenchmark(
        model_path=args.model_dir,
        input_shape=(1, 3, 224, 224),  # Example shape for classification
        warmup_runs=args.warmup_runs
    )

    try:
        # Run benchmarks and save results
        benchmark.save_results(args.output)
        print(f"Benchmark results saved to {args.output}")
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return 1

if __name__ == '__main__':
    main()
