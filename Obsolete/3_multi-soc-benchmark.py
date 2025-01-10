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

SOC_CONFIGS = {
    "rk3588s": SoCInfo(
        name="RK3588S",
        accelerator_type=AcceleratorType.RKNN,
        compute_capability=6.0,  # 6 TOPS
        memory_type="LPDDR4",
        memory_size=8192,
        requires_quantization=True,
        supported_formats=["onnx", "caffe", "tensorflow"]
    ),
    "imx93": SoCInfo(
        name="i.MX 93",
        accelerator_type=AcceleratorType.TENSORRT,
        compute_capability=2.0,  # 2 TOPS
        memory_type="LPDDR4",
        memory_size=2048,
        requires_quantization=True,
        supported_formats=["onnx", "tensorflow"]
    ),
    "rk3399": SoCInfo(
        name="RK3399",
        accelerator_type=AcceleratorType.TFLITE,
        compute_capability=0.096,  # 96 GOPS (NEON)
        memory_type="LPDDR4",
        memory_size=4096,
        requires_quantization=False,
        supported_formats=["tflite"]
    ),
    "bcm2711": SoCInfo(
        name="BCM2711",
        accelerator_type=AcceleratorType.TFLITE,
        compute_capability=0.032,  # 32 GOPS
        memory_type="LPDDR4",
        memory_size=8192,
        requires_quantization=False,
        supported_formats=["tflite"]
    ),
    "spacemit_k1": SoCInfo(
        name="SpacemiT K1",
        accelerator_type=AcceleratorType.CPU,  # Need to update when SDK available
        compute_capability=1.0,
        memory_type="LPDDR4",
        memory_size=4096,
        requires_quantization=True,
        supported_formats=["onnx"]
    ),
    "am67a": SoCInfo(
        name="AM67A",
        accelerator_type=AcceleratorType.TIDL,
        compute_capability=8.0,  # 8 TOPS
        memory_type="LPDDR4",
        memory_size=8192,
        requires_quantization=True,
        supported_formats=["onnx", "tflite"]
    ),
    "rp2350": SoCInfo(
        name="RP2350",
        accelerator_type=AcceleratorType.CPU,
        compute_capability=0.01,  # Estimated GOPS
        memory_type="SRAM",
        memory_size=264,  # 264KB
        requires_quantization=True,
        supported_formats=["tflite-micro"]
    )
}

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

class RKNNRunner(ModelRunner):
    def prepare_model(self):
        from rknn.api import RKNN
        self.rknn = RKNN()
        # RKNN specific initialization
        pass

    def run_inference(self, input_data: np.ndarray) -> float:
        start = time.perf_counter()
        self.rknn.inference(inputs=[input_data])
        end = time.perf_counter()
        return (end - start) * 1000

class TFLiteRunner(ModelRunner):
    def prepare_model(self):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        # Enable NEON acceleration if available
        if self.soc_info.name == "RK3399":
            try:
                self.interpreter.set_num_threads(6)  # RK3399 has 6 cores
            except:
                pass

    def run_inference(self, input_data: np.ndarray) -> float:
        start = time.perf_counter()
        self.interpreter.set_tensor(
            self.interpreter.get_input_details()[0]['index'], 
            input_data
        )
        self.interpreter.invoke()
        end = time.perf_counter()
        return (end - start) * 1000

class TensorRTRunner(ModelRunner):
    def prepare_model(self):
        import tensorrt as trt
        # TensorRT specific initialization for i.MX 93
        pass

    def run_inference(self, input_data: np.ndarray) -> float:
        start = time.perf_counter()
        # TensorRT inference
        end = time.perf_counter()
        return (end - start) * 1000

class TIDLRunner(ModelRunner):
    def prepare_model(self):
        # TI Deep Learning specific initialization
        pass

    def run_inference(self, input_data: np.ndarray) -> float:
        start = time.perf_counter()
        # TIDL inference
        end = time.perf_counter()
        return (end - start) * 1000

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

class MultiSoCBenchmark:
    def __init__(self, 
                 model_path: str,
                 soc_type: str,
                 input_shape: tuple,
                 warmup_runs: int = 5):
        self.model_path = Path(model_path)
        self.soc_info = SOC_CONFIGS[soc_type]
        self.input_shape = input_shape
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
        self.logger = self._setup_logging()

        # Select appropriate runner based on accelerator type
        runner_classes = {
            AcceleratorType.RKNN: RKNNRunner,
            AcceleratorType.TFLITE: TFLiteRunner,
            AcceleratorType.TENSORRT: TensorRTRunner,
            AcceleratorType.TIDL: TIDLRunner
        }
        
        runner_class = runner_classes.get(self.soc_info.accelerator_type)
        if runner_class:
            self.runner = runner_class(str(model_path), self.soc_info)
        else:
            raise ValueError(f"Unsupported accelerator type: {self.soc_info.accelerator_type}")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"MultiSoCBenchmark-{self.soc_info.name}")
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
        """Measure power usage using platform-specific methods"""
        # Different power measurement paths for different boards
        power_paths = {
            "rk3588s": "/sys/class/power_supply/battery/power_now",
            "imx93": "/sys/class/power_supply/batt/power_now",
            "am67a": "/sys/class/power_supply/batt/power_now",
            # Add more paths as needed
        }
        
        path = power_paths.get(self.soc_info.name.lower())
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return float(f.read()) / 1000000.0  # convert μW to W
            except:
                pass
        return None

    def _measure_temperature(self) -> Optional[float]:
        """Measure SoC temperature"""
        temp_paths = {
            "rk3588s": "/sys/class/thermal/thermal_zone0/temp",
            "bcm2711": "/sys/class/thermal/thermal_zone0/temp",
            "am67a": "/sys/class/thermal/thermal_zone1/temp",
            # Add more paths as needed
        }
        
        path = temp_paths.get(self.soc_info.name.lower())
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return float(f.read()) / 1000.0  # convert mC to C
            except:
                pass
        return None

    def benchmark_model(self) -> BenchmarkResult:
        """Run complete benchmark for the model"""
        try:
            self.runner.prepare_model()
            
            # Create dummy input data
            input_data = np.random.rand(*self.input_shape).astype(np.float32)
            
            # Warmup runs
            self.logger.info("Performing warmup runs...")
            for _ in range(self.warmup_runs):
                self.runner.run_inference(input_data)

            # Actual benchmark runs
            times = []
            memory_usage = []
            power_readings = []
            cpu_usage = []
            temperature_readings = []

            for i in range(100):  # 100 inference runs
                if i % 10 == 0:
                    self.logger.info(f"Running inference {i}/100")
                
                memory_usage.append(self._measure_memory())
                power = self._measure_power()
                if power:
                    power_readings.append(power)
                
                cpu_percent = psutil.cpu_percent()
                cpu_usage.append(cpu_percent)
                
                temp = self._measure_temperature()
                if temp:
                    temperature_readings.append(temp)
                
                inference_time = self.runner.run_inference(input_data)
                times.append(inference_time)

            avg_time = np.mean(times)
            avg_memory = np.mean(memory_usage)
            avg_power = np.mean(power_readings) if power_readings else None
            avg_cpu = np.mean(cpu_usage)
            avg_temp = np.mean(temperature_readings) if temperature_readings else None
            throughput = 1000 / avg_time  # convert ms to inferences/second

            result = BenchmarkResult(
                soc_name=self.soc_info.name,
                model_name=self.model_path.stem,
                accelerator_type=self.soc_info.accelerator_type.value,
                inference_time=avg_time,
                memory_usage=avg_memory,
                power_usage=avg_power,
                throughput=throughput,
                quantization=None,  # Set based on model analysis
                input_shape=self.input_shape,
                cpu_usage=avg_cpu,
                temperature=avg_temp
            )
            
            self.results.append(result)
            return result

        except Exception as e:
            self.logger.error(f"Error benchmarking {self.model_path}: {str(e)}")
            raise

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file"""
        results_dict = {
            "soc_info": {
                "name": self.soc_info.name,
                "accelerator_type": self.soc_info.accelerator_type.value,
                "compute_capability": self.soc_info.compute_capability,
                "memory_type": self.soc_info.memory_type,
                "memory_size": self.soc_info.memory_size
            },
            "results": [vars(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Multi-SoC AI Benchmark')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--soc-type', type=str, required=True,
                       choices=list(SOC_CONFIGS.keys()),
                       help='SoC type')
    parser.add_argument('--input-shape', type=str, required=True,
                       help='Input shape as comma-separated values (e.g., 1,3,224,224)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file path for results')
    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))

    benchmark = MultiSoCBenchmark(
        model_path=args.model_path,
        soc_type=args.soc_type,
        input_shape=input_shape
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
