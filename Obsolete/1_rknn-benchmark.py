#!/usr/bin/env python3
import os
import time
import psutil
import argparse
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from rknn.api import RKNN

@dataclass
class RKNNBenchmarkResult:
    model_name: str
    inference_time: float      # in milliseconds
    memory_usage: float        # in MB
    power_usage: Optional[float]  # in watts, if available
    throughput: float         # inferences per second
    quantization: str         # 'i8' or 'fp'
    input_shape: tuple
    platform: str            # 'rv1106' or 'rv1103'

class RKNNBenchmark:
    def __init__(self, 
                 onnx_path: str, 
                 platform: str,
                 quantization: str = 'i8',
                 warmup_runs: int = 5):
        self.onnx_path = Path(onnx_path)
        self.platform = platform
        self.quantization = quantization
        self.warmup_runs = warmup_runs
        self.results: List[RKNNBenchmarkResult] = []
        self.logger = self._setup_logging()
        self.rknn = RKNN()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("RKNNBenchmark")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _convert_model(self) -> str:
        """Convert ONNX model to RKNN format"""
        self.logger.info(f"Converting {self.onnx_path} to RKNN format")
        
        # Set target platform
        self.rknn.config(target_platform=[self.platform])
        
        # Load ONNX model
        ret = self.rknn.load_onnx(model=str(self.onnx_path))
        if ret != 0:
            raise Exception("Load ONNX model failed!")

        # Build RKNN model
        if self.quantization == 'i8':
            ret = self.rknn.build(do_quantization=True, dataset='dataset.txt')
        else:
            ret = self.rknn.build(do_quantization=False)
        
        if ret != 0:
            raise Exception("Build RKNN model failed!")

        # Export RKNN model
        rknn_path = self.onnx_path.with_suffix('.rknn')
        ret = self.rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise Exception("Export RKNN model failed!")

        return str(rknn_path)

    def _measure_memory(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _measure_power(self) -> Optional[float]:
        """
        Attempt to measure power usage on RV1103/RV1106
        Returns power in watts or None if not available
        """
        try:
            # Try to read from sysfs power monitoring
            with open('/sys/class/power_supply/bat/power_now', 'r') as f:
                return float(f.read()) / 1000000.0  # convert μW to W
        except:
            return None

    def run_inference(self, input_data: np.ndarray) -> float:
        """Run a single inference and return time in milliseconds"""
        start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        end = time.perf_counter()
        return (end - start) * 1000  # convert to ms

    def benchmark_model(self) -> RKNNBenchmarkResult:
        """Run complete benchmark for the model"""
        try:
            # Convert model if needed
            rknn_path = self._convert_model()
            self.logger.info(f"Using RKNN model: {rknn_path}")
            
            # Get model input shape
            input_shape = self.rknn.inputs[0].shape
            
            # Create dummy input data
            input_data = np.random.rand(*input_shape).astype(np.float32)
            
            # Warmup runs
            self.logger.info("Performing warmup runs...")
            for _ in range(self.warmup_runs):
                self.run_inference(input_data)

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
                
                inference_time = self.run_inference(input_data)
                times.append(inference_time)

            avg_time = np.mean(times)
            avg_memory = np.mean(memory_usage)
            avg_power = np.mean(power_readings) if power_readings else None
            throughput = 1000 / avg_time  # convert ms to inferences/second

            result = RKNNBenchmarkResult(
                model_name=self.onnx_path.stem,
                inference_time=avg_time,
                memory_usage=avg_memory,
                power_usage=avg_power,
                throughput=throughput,
                quantization=self.quantization,
                input_shape=input_shape,
                platform=self.platform
            )
            
            self.results.append(result)
            return result

        except Exception as e:
            self.logger.error(f"Error benchmarking {self.onnx_path}: {str(e)}")
            raise
        finally:
            self.rknn.release()

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file"""
        results_dict = {
            "platform_info": {
                "cpu_info": self._get_cpu_info(),
                "total_memory": psutil.virtual_memory().total / (1024 * 1024),  # MB
                "os_info": self._get_os_info(),
                "target_platform": self.platform
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
    parser = argparse.ArgumentParser(description='RKNN Benchmark for RV1103/RV1106')
    parser.add_argument('--onnx-model', type=str, required=True,
                       help='Path to ONNX model')
    parser.add_argument('--platform', type=str, required=True,
                       choices=['rv1103', 'rv1106'],
                       help='Target platform')
    parser.add_argument('--quantization', type=str, default='i8',
                       choices=['i8', 'fp'],
                       help='Quantization type (i8 or fp)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file path for results')
    parser.add_argument('--warmup-runs', type=int, default=5,
                       help='Number of warmup runs before benchmarking')
    args = parser.parse_args()

    benchmark = RKNNBenchmark(
        onnx_path=args.onnx_model,
        platform=args.platform,
        quantization=args.quantization,
        warmup_runs=args.warmup_runs
    )

    try:
        benchmark.benchmark_model()
        benchmark.save_results(args.output)
        print(f"Benchmark results saved to {args.output}")
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return 1

if __name__ == '__main__':
    main()
