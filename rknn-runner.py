"""
RKNN Benchmark Runner
Supports RV1103/RV1106/RK3588 with standard models
"""
import os
import json
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
from rknn.api import RKNN

@dataclass
class ModelConfig:
    name: str
    path: str
    input_shape: tuple
    quantization: str = "int8"

class RKNNBenchmark:
    # Platform configurations
    PLATFORM_CONFIGS = {
        "rv1103": {
            "target": "rv1103",
            "optimization": 3,
            "core_mask": 1,
            "models": [
                ModelConfig("mobilenetv2", "models/mobilenetv2-12.onnx", (1, 3, 224, 224)),
                ModelConfig("yolov5s", "models/yolov5s.onnx", (1, 3, 640, 640))
            ]
        },
        "rv1106": {
            "target": "rv1106",
            "optimization": 3,
            "core_mask": 1,
            "models": [
                ModelConfig("mobilenetv2", "models/mobilenetv2-12.onnx", (1, 3, 224, 224)),
                ModelConfig("yolov5s", "models/yolov5s.onnx", (1, 3, 640, 640))
            ]
        },
        "rk3588": {
            "target": "rk3588",
            "optimization": 3,
            "core_mask": None,  # Uses all available cores
            "models": [
                ModelConfig("mobilenetv2", "models/mobilenetv2-12.onnx", (1, 3, 224, 224)),
                ModelConfig("yolov5s", "models/yolov5s.onnx", (1, 3, 640, 640)),
                ModelConfig("yolov8n", "models/yolov8n.onnx", (1, 3, 640, 640))
            ]
        }
    }

    def __init__(self):
        self.platform = self._detect_platform()
        self.config = self.PLATFORM_CONFIGS.get(self.platform)
        self.rknn = RKNN(verbose=True)
        
    def _detect_platform(self) -> str:
        """Detect RKNN platform"""
        # TODO: Implement proper detection
        # For now, let user specify or default to RV1103
        return os.environ.get("RKNN_PLATFORM", "rv1103")

    def _prepare_model(self, model_config: ModelConfig) -> bool:
        """Prepare RKNN model"""
        try:
            print(f"\nPreparing model: {model_config.name}")
            
            # RKNN Configuration
            self.rknn.config(
                target_platform=self.config["target"],
                optimization_level=self.config["optimization"],
                core_mask=self.config["core_mask"],
                quantize_input_node=(model_config.quantization != "none")
            )
            
            # Load model
            ret = self.rknn.load_onnx(model_config.path)
            if ret != 0:
                print('Load model failed!')
                return False
                
            # Build model
            ret = self.rknn.build(do_quantization=(model_config.quantization != "none"))
            if ret != 0:
                print('Build model failed!')
                return False
                
            # Init runtime
            ret = self.rknn.init_runtime()
            if ret != 0:
                print('Init runtime failed!')
                return False
                
            return True
            
        except Exception as e:
            print(f"Error preparing model: {str(e)}")
            return False

    def _benchmark_model(self, model_config: ModelConfig, 
                        num_warmup: int = 5, 
                        num_runs: int = 100) -> Dict:
        """Run benchmark for a specific model"""
        try:
            # Create input data
            input_data = np.random.rand(*model_config.input_shape).astype(np.float32)
            
            # Warmup runs
            print("Performing warmup runs...")
            for _ in range(num_warmup):
                self.rknn.inference(inputs=[input_data])
            
            # Benchmark runs
            print(f"Running {num_runs} iterations...")
            times = []
            
            for i in range(num_runs):
                if i % 10 == 0:
                    print(f"Progress: {i}/{num_runs}")
                    
                start = time.perf_counter()
                self.rknn.inference(inputs=[input_data])
                times.append((time.perf_counter() - start) * 1000)  # Convert to ms
                
            # Calculate statistics
            times = np.array(times)
            
            return {
                "name": model_config.name,
                "format": "onnx",
                "input_shape": list(model_config.input_shape),
                "quantization": model_config.quantization,
                "performance": {
                    "inference_ms": {
                        "min": float(np.min(times)),
                        "max": float(np.max(times)),
                        "avg": float(np.mean(times))
                    },
                    "throughput": {
                        "fps": float(1000 / np.mean(times))
                    },
                    "memory_usage_kb": self._get_memory_usage()
                }
            }
            
        except Exception as e:
            print(f"Error benchmarking model: {str(e)}")
            return None

    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss // 1024  # Convert to KB
        except:
            return 0

    def _get_device_info(self) -> Dict:
        """Get device information"""
        # TODO: Implement proper detection of processor details
        platform_info = {
            "rv1103": {
                "processor_name": "RV1103",
                "freq_mhz": 1008,
                "compute": "0.5 TOPS"
            },
            "rv1106": {
                "processor_name": "RV1106",
                "freq_mhz": 1008,
                "compute": "0.5 TOPS"
            },
            "rk3588": {
                "processor_name": "RK3588",
                "freq_mhz": 2400,
                "compute": "6.0 TOPS"
            }
        }
        
        info = platform_info.get(self.platform, {})
        
        return {
            "name": f"RKNN {info.get('processor_name', 'Unknown')}",
            "type": "SBC",
            "processor": {
                "name": info.get('processor_name', 'Unknown'),
                "architecture": "ARM64",
                "frequency_mhz": info.get('freq_mhz', 0),
                "features": ["neon", "simd"]
            },
            "accelerator": {
                "name": "RKNN NPU",
                "type": "NPU",
                "compute_capability": info.get('compute', 'Unknown'),
                "supported_formats": ["onnx", "caffe", "darknet"]
            },
            "memory": {
                "total_kb": self._get_total_memory(),
                "type": "DDR3"
            }
        }

    def _get_total_memory(self) -> int:
        """Get total system memory in KB"""
        try:
            import psutil
            return int(psutil.virtual_memory().total / 1024)
        except:
            return 0

    def run_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        results = {
            "device": self._get_device_info(),
            "benchmarks": {
                "matrix_ops": None,  # Matrix ops not supported in RKNN mode
                "models": []
            }
        }

        if not self.config:
            print(f"Unsupported platform: {self.platform}")
            return results

        for model_config in self.config["models"]:
            print(f"\nBenchmarking {model_config.name}...")
            
            # Reset RKNN instance
            if hasattr(self, 'rknn'):
                self.rknn.release()
            self.rknn = RKNN(verbose=True)
            
            # Prepare and benchmark model
            if self._prepare_model(model_config):
                model_results = self._benchmark_model(model_config)
                if model_results:
                    results["benchmarks"]["models"].append(model_results)
            
        return results

    def cleanup(self):
        """Cleanup RKNN runtime"""
        if hasattr(self, 'rknn'):
            self.rknn.release()

def main():
    benchmark = RKNNBenchmark()
    try:
        results = benchmark.run_benchmark()
        
        # Save results
        output_file = "rknn_benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        return results
    finally:
        benchmark.cleanup()

if __name__ == '__main__':
    main()