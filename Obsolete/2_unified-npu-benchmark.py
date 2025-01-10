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
from typing import List, Dict, Optional, Union
from enum import Enum

class NPUType(Enum):
    RKNN = "rknn"
    CVITEK = "cvitek"

@dataclass
class ModelInfo:
    model_path: str
    input_shape: tuple
    mean: List[float]
    scale: List[float]
    pixel_format: str  # "rgb" or "bgr"
    quantization: str  # "i8" or "fp" for RKNN, "INT8" or "BF16" for CVITEK

@dataclass
class BenchmarkResult:
    model_name: str
    npu_type: NPUType
    inference_time: float      # in milliseconds
    memory_usage: float        # in MB
    power_usage: Optional[float]  # in watts, if available
    throughput: float         # inferences per second
    quantization: str
    input_shape: tuple

class UnifiedNPUBenchmark:
    def __init__(self, 
                 model_info: ModelInfo,
                 platform: str,
                 npu_type: NPUType,
                 warmup_runs: int = 5):
        self.model_info = model_info
        self.platform = platform
        self.npu_type = npu_type
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
        self.logger = self._setup_logging()

        if npu_type == NPUType.RKNN:
            from rknn.api import RKNN
            self.rknn = RKNN()
        else:  # CVITEK
            import torch
            self.torch = torch

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("UnifiedNPUBenchmark")
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
        """Measure power usage on supported platforms"""
        power_paths = [
            '/sys/class/power_supply/BAT0/power_now',  # Generic Linux
            '/sys/class/ampower/power_now',            # CV1800B specific
            '/sys/class/rk_power/power_now'            # RV1103/RV1106 specific
        ]
        
        for path in power_paths:
            try:
                with open(path, 'r') as f:
                    return float(f.read()) / 1000000.0  # convert μW to W
            except:
                continue
        return None

    def _convert_rknn_model(self) -> str:
        """Convert model to RKNN format"""
        self.logger.info("Converting model to RKNN format...")
        
        # Load and convert model
        ret = self.rknn.config(mean_values=[self.model_info.mean],
                             std_values=[self.model_info.scale],
                             target_platform=self.platform)
        if ret != 0:
            raise Exception("RKNN config failed")

        ret = self.rknn.load_onnx(model=self.model_info.model_path)
        if ret != 0:
            raise Exception("Load ONNX model failed")

        ret = self.rknn.build(do_quantization=self.model_info.quantization == "i8")
        if ret != 0:
            raise Exception("Build RKNN model failed")

        rknn_path = str(Path(self.model_info.model_path).with_suffix('.rknn'))
        ret = self.rknn.export_rknn(rknn_path)
        if ret != 0:
            raise Exception("Export RKNN model failed")

        return rknn_path

    def _convert_cvitek_model(self) -> str:
        """Convert model to CVITEK format using TPU-MLIR"""
        self.logger.info("Converting model to CVITEK format...")
        
        # Assuming model_transform.py and model_deploy.py are in PATH
        mlir_path = str(Path(self.model_info.model_path).with_suffix('.mlir'))
        cvimodel_path = str(Path(self.model_info.model_path).with_suffix('.cvimodel'))
        
        # Convert to MLIR
        os.system(f"""model_transform.py \\
            --model_name {Path(self.model_info.model_path).stem} \\
            --model_def {self.model_info.model_path} \\
            --input_shapes {self.model_info.input_shape} \\
            --mean {','.join(map(str, self.model_info.mean))} \\
            --scale {','.join(map(str, self.model_info.scale))} \\
            --pixel_format {self.model_info.pixel_format} \\
            --mlir {mlir_path}
        """)

        # Generate calibration table for INT8
        if self.model_info.quantization == "INT8":
            os.system(f"""run_calibration.py {mlir_path} \\
                --dataset ./dataset \\
                --input_num 100 \\
                -o {Path(mlir_path).stem}_cali_table
            """)

        # Convert to cvimodel
        quant_flag = "--quantize INT8" if self.model_info.quantization == "INT8" else ""
        os.system(f"""model_deploy.py \\
            --mlir {mlir_path} \\
            {quant_flag} \\
            --chip {self.platform} \\
            --model {cvimodel_path}
        """)

        return cvimodel_path

    def benchmark_model(self) -> BenchmarkResult:
        """Run complete benchmark for the model"""
        try:
            # Convert model if needed
            if self.npu_type == NPUType.RKNN:
                model_path = self._convert_rknn_model()
                self.rknn.init_runtime()
            else:  # CVITEK
                model_path = self._convert_cvitek_model()
                # Load cvimodel using appropriate SDK

            # Create dummy input data
            input_data = np.random.rand(*self.model_info.input_shape).astype(np.float32)
            
            # Warmup runs
            self.logger.info("Performing warmup runs...")
            for _ in range(self.warmup_runs):
                if self.npu_type == NPUType.RKNN:
                    self.rknn.inference(inputs=[input_data])
                else:  # CVITEK
                    pass  # Use appropriate inference call

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
                
                start = time.perf_counter()
                if self.npu_type == NPUType.RKNN:
                    self.rknn.inference(inputs=[input_data])
                else:  # CVITEK
                    pass  # Use appropriate inference call
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms

            avg_time = np.mean(times)
            avg_memory = np.mean(memory_usage)
            avg_power = np.mean(power_readings) if power_readings else None
            throughput = 1000 / avg_time  # Convert ms to inferences/second

            result = BenchmarkResult(
                model_name=Path(self.model_info.model_path).stem,
                npu_type=self.npu_type,
                inference_time=avg_time,
                memory_usage=avg_memory,
                power_usage=avg_power,
                throughput=throughput,
                quantization=self.model_info.quantization,
                input_shape=self.model_info.input_shape
            )
            
            self.results.append(result)
            return result

        except Exception as e:
            self.logger.error(f"Error benchmarking {self.model_info.model_path}: {str(e)}")
            raise
        finally:
            if self.npu_type == NPUType.RKNN:
                self.rknn.release()

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file"""
        results_dict = {
            "platform_info": {
                "cpu_info": self._get_cpu_info(),
                "total_memory": psutil.virtual_memory().total / (1024 * 1024),  # MB
                "os_info": self._get_os_info(),
                "npu_type": self.npu_type.value,
                "platform": self.platform
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
    parser = argparse.ArgumentParser(description='Unified NPU Benchmark for Resource-Constrained SBCs')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model file (ONNX or Caffe)')
    parser.add_argument('--platform', type=str, required=True,
                       choices=['rv1106', 'rv1103', 'cv180x', 'sg200x'],
                       help='Target platform')
    parser.add_argument('--npu-type', type=str, required=True,
                       choices=['rknn', 'cvitek'],
                       help='NPU type')
    parser.add_argument('--quantization', type=str, default='i8',
                       choices=['i8', 'fp', 'INT8', 'BF16'],
                       help='Quantization type')
    parser.add_argument('--input-shape', type=str, required=True,
                       help='Input shape as comma-separated values (e.g., 1,3,224,224)')
    parser.add_argument('--mean', type=str, required=True,
                       help='Mean values as comma-separated floats')
    parser.add_argument('--scale', type=str, required=True,
                       help='Scale values as comma-separated floats')
    parser.add_argument('--pixel-format', type=str, default='rgb',
                       choices=['rgb', 'bgr'],
                       help='Pixel format')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file path for results')
    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Parse mean and scale
    mean = list(map(float, args.mean.split(',')))
    scale = list(map(float, args.scale.split(',')))

    model_info = ModelInfo(
        model_path=args.model_path,
        input_shape=input_shape,
        mean=mean,
        scale=scale,
        pixel_format=args.pixel_format,
        quantization=args.quantization
    )

    benchmark = UnifiedNPUBenchmark(
        model_info=model_info,
        platform=args.platform,
        npu_type=NPUType[args.npu_type.upper()],
        warmup_runs=5
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
