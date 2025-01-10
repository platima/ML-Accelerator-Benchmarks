#!/usr/bin/env python3
"""
SoC Benchmark Runner
Unified benchmark tool for various SoCs with ML/AI acceleration capabilities.
Supports:
- RV1103/RV1106 (RKNN)
- CV1800B/SG2002 (CVITEK/TPU-MLIR)
- RK3588S (RKNN)
- i.MX 93 (TensorRT)
- AM67A (TIDL)
- RK3399 (NEON)
- BCM2711 (CPU)
- SpacemiT K1
- RP2350
"""

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

class AcceleratorType(Enum):
    RKNN = "rknn"           # RV1103/RV1106/RK3588S
    CVITEK = "cvitek"       # CV1800B/SG2002
    TFLITE = "tflite"       # Generic TFLite for CPU/NEON
    TENSORRT = "tensorrt"   # NXP i.MX 93
    TIDL = "tidl"           # TI AM67A
    CPU = "cpu"             # Fallback for no acceleration

@dataclass
class SoCInfo:
    name: str
    accelerator_type: AcceleratorType
    compute_capability: float  # TOPS or GOPS
    memory_type: str
    memory_size: int          # in MB
    requires_quantization: bool
    supported_formats: List[str]
    power_path: Optional[str] = None
    temp_path: Optional[str] = None

# SoC configurations database
SOC_CONFIGS = {
    "rv1106": SoCInfo(
        name="RV1106",
        accelerator_type=AcceleratorType.RKNN,
        compute_capability=0.5,  # 0.5 TOPS
        memory_type="DDR3",
        memory_size=64,
        requires_quantization=True,
        supported_formats=["onnx", "caffe"],
        power_path="/sys/class/power_supply/bat/power_now",
        temp_path="/sys/class/thermal/thermal_zone0/temp"
    ),
    "cv1800b": SoCInfo(
        name="CV1800B",
        accelerator_type=AcceleratorType.CVITEK,
        compute_capability=0.5,  # 0.5 TOPS
        memory_type="DDR3",
        memory_size=64,
        requires_quantization=True,
        supported_formats=["onnx", "caffe"],
        power_path="/sys/class/ampower/power_now",
        temp_path="/sys/class/thermal/thermal_zone0/temp"
    ),
    "rk3588s": SoCInfo(
        name="RK3588S",
        accelerator_type=AcceleratorType.RKNN,
        compute_capability=6.0,  # 6 TOPS
        memory_type="LPDDR4",
        memory_size=8192,
        requires_quantization=True,
        supported_formats=["onnx", "caffe", "tensorflow"],
        power_path="/sys/class/power_supply/battery/power_now",
        temp_path="/sys/class/thermal/thermal_zone0/temp"
    ),
    # Add other SoCs here...
}

@dataclass
class BenchmarkResult:
    soc_name: str
    model_name: str
    accelerator_type: str
    inference_time: float      # in milliseconds
    memory_usage: float        # in MB
    power_usage: Optional[float]  # in watts
    throughput: float          # inferences/second
    quantization: Optional[str]
    input_shape: tuple
    cpu_usage: float          # percentage
    temperature: Optional[float]  # in Celsius
    timestamp: str            # ISO format timestamp
    batch_size: int

class ModelRunner:
    """Base class for model execution"""
    def __init__(self, model_path: str, soc_info: SoCInfo):
        self.model_path = model_path
        self.soc_info = soc_info
        self.logger = logging.getLogger(f"ModelRunner-{soc_info.name}")

    def prepare_model(self):
        raise NotImplementedError()

    def run_inference(self, input_data: np.ndarray) -> float:
        raise NotImplementedError()

    def cleanup(self):
        pass

class RKNNRunner(ModelRunner):
    def prepare_model(self):
        try:
            from rknn.api import RKNN
            self.rknn = RKNN()
            # RKNN specific initialization
            return True
        except ImportError:
            self.logger.error("RKNN toolkit not installed")
            return False

    def run_inference(self, input_data: np.ndarray) -> float:
        start = time.perf_counter()
        self.rknn.inference(inputs=[input_data])
        end = time.perf_counter()
        return (end - start) * 1000

    def cleanup(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()

class CVITEKRunner(ModelRunner):
    def prepare_model(self):
        # CVITEK/TPU-MLIR specific initialization
        try:
            import cvi_toolkit
            self.cvi = cvi_toolkit
            return True
        except ImportError:
            self.logger.error("CVITEK toolkit not installed")
            return False

    def run_inference(self, input_data: np.ndarray) -> float:
        start = time.perf_counter()
        # CVITEK inference
        end = time.perf_counter()
        return (end - start) * 1000

class Benchmark:
    def __init__(self, 
                 model_path: str,
                 soc_type: str,
                 input_shape: tuple,
                 batch_size: int = 1,
                 warmup_runs: int = 5,
                 num_runs: int = 100):
        self.model_path = Path(model_path)
        self.soc_info = SOC_CONFIGS[soc_type]
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        self.logger = self._setup_logging()

        # Select appropriate runner
        runners = {
            AcceleratorType.RKNN: RKNNRunner,
            AcceleratorType.CVITEK: CVITEKRunner,
        }
        runner_class = runners.get(self.soc_info.accelerator_type)
        if runner_class:
            self.runner = runner_class(str(model_path), self.soc_info)
        else:
            raise ValueError(f"Unsupported accelerator type: {self.soc_info.accelerator_type}")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"Benchmark-{self.soc_info.name}")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _measure_system_metrics(self) -> Dict[str, float]:
        """Measure various system metrics"""
        metrics = {
            "memory_usage": psutil.Process().memory_info().rss / (1024 * 1024),
            "cpu_usage": psutil.cpu_percent(),
            "power_usage": None,
            "temperature": None
        }

        # Measure power if available
        if self.soc_info.power_path and os.path.exists(self.soc_info.power_path):
            try:
                with open(self.soc_info.power_path, 'r') as f:
                    metrics["power_usage"] = float(f.read()) / 1000000.0
            except:
                pass

        # Measure temperature if available
        if self.soc_info.temp_path and os.path.exists(self.soc_info.temp_path):
            try:
                with open(self.soc_info.temp_path, 'r') as f:
                    metrics["temperature"] = float(f.read()) / 1000.0
            except:
                pass

        return metrics

    def run(self) -> BenchmarkResult:
        """Run the benchmark"""
        try:
            if not self.runner.prepare_model():
                raise RuntimeError("Failed to prepare model")

            # Create input data
            input_data = np.random.rand(*self.input_shape).astype(np.float32)

            # Warmup runs
            self.logger.info("Performing warmup runs...")
            for _ in range(self.warmup_runs):
                self.runner.run_inference(input_data)

            # Benchmark runs
            self.logger.info(f"Running {self.num_runs} benchmark iterations...")
            times = []
            metrics_list = []

            for i in range(self.num_runs):
                if i % 10 == 0:
                    self.logger.info(f"Progress: {i}/{self.num_runs}")

                metrics_list.append(self._measure_system_metrics())
                times.append(self.runner.run_inference(input_data))

            # Calculate averages
            avg_time = np.mean(times)
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list if m[k] is not None])
                for k in metrics_list[0].keys()
            }

            result = BenchmarkResult(
                soc_name=self.soc_info.name,
                model_name=self.model_path.stem,
                accelerator_type=self.soc_info.accelerator_type.value,
                inference_time=avg_time,
                memory_usage=avg_metrics["memory_usage"],
                power_usage=avg_metrics.get("power_usage"),
                throughput=1000 / avg_time,
                quantization=None,  # Set based on model analysis
                input_shape=self.input_shape,
                cpu_usage=avg_metrics["cpu_usage"],
                temperature=avg_metrics.get("temperature"),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                batch_size=self.batch_size
            )

            return result

        finally:
            self.runner.cleanup()

    def save_result(self, result: BenchmarkResult, output_path: str):
        """Save benchmark results to JSON file"""
        data = {
            "soc_info": {
                "name": self.soc_info.name,
                "accelerator_type": self.soc_info.accelerator_type.value,
                "compute_capability": self.soc_info.compute_capability,
                "memory_type": self.soc_info.memory_type,
                "memory_size": self.soc_info.memory_size
            },
            "model_info": {
                "name": self.model_path.stem,
                "path": str(self.model_path),
                "input_shape": self.input_shape
            },
            "benchmark_settings": {
                "warmup_runs": self.warmup_runs,
                "num_runs": self.num_runs,
                "batch_size": self.batch_size
            },
            "results": vars(result)
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='SoC ML/AI Benchmark Tool')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--soc-type', type=str, required=True,
                       choices=list(SOC_CONFIGS.keys()),
                       help='SoC type')
    parser.add_argument('--input-shape', type=str, required=True,
                       help='Input shape as comma-separated values (e.g., 1,3,224,224)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--warmup-runs', type=int, default=5,
                       help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file path for results')
    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))

    benchmark = Benchmark(
        model_path=args.model_path,
        soc_type=args.soc_type,
        input_shape=input_shape,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs
    )

    try:
        result = benchmark.run()
        benchmark.save_result(result, args.output)
        print(f"Benchmark results saved to {args.output}")
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return 1

if __name__ == '__main__':
    main()