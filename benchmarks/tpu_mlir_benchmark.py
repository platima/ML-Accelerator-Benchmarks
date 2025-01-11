"""
TPU/NPU Benchmark Implementation
Handles benchmarking for RKNN and CVITEK platforms
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import time

from .base import BenchmarkBase, BenchmarkConfig
from runners import TPURunner, RunnerConfig

class TPUBenchmark(BenchmarkBase):
    """TPU/NPU benchmark implementation"""
    
    # Standard model configurations
    MODEL_CONFIGS = {
        "mobilenetv2": {
            "input_shape": (1, 3, 224, 224),
            "pixel_format": "rgb",
            "mean": (0.0, 0.0, 0.0),
            "scale": (0.017, 0.017, 0.017),  # 1/58.8235294
            "quantization": "int8"
        },
        "yolov5s": {
            "input_shape": (1, 3, 640, 640),
            "pixel_format": "rgb",
            "mean": (0.0, 0.0, 0.0),
            "scale": (1/255.0, 1/255.0, 1/255.0),
            "quantization": "int8"
        }
    }

    def __init__(self, 
                 model_name: str,
                 model_path: str,
                 test_images: List[str],
                 config: Optional[BenchmarkConfig] = None):
        if config is None:
            # Use default configuration for the model
            model_config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["mobilenetv2"])
            runner_config = RunnerConfig(
                input_shape=model_config["input_shape"],
                quantization=model_config["quantization"],
                model_path=model_path
            )
            config = BenchmarkConfig(
                name=f"TPU_{model_name}",
                runner_config=runner_config
            )
        super().__init__(config)
        
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.test_images = test_images
        self.runner = None

    def initialize(self) -> bool:
        """Initialize TPU runner and resources"""
        try:
            self.runner = TPURunner(self.config.runner_config)
            return True
        except Exception as e:
            print(f"Failed to initialize TPU benchmark: {e}")
            return False

    def run(self) -> Dict:
        """Run complete benchmark suite"""
        if not self.runner:
            raise RuntimeError("TPU runner not initialized")
            
        results = {
            "model_name": self.model_name,
            "model_config": self.MODEL_CONFIGS.get(self.model_name, {}),
            "hardware_info": self._get_hardware_info(),
            "timings": {
                "inference_ms": [],
                "pre_process_ms": [],
                "post_process_ms": [],
                "total_ms": []
            },
            "memory_usage_kb": [],
            "extra_metrics": []
        }
        
        # Warmup runs
        print(f"\nPerforming {self.config.warmup_runs} warmup runs...")
        for _ in range(self.config.warmup_runs):
            self._run_single(self.test_images[0])
            
        # Benchmark runs
        print(f"\nPerforming {self.config.num_runs} benchmark runs...")
        for i in range(self.config.num_runs):
            if i % 10 == 0:
                print(f"Progress: {i}/{self.config.num_runs}")
                
            # Cycle through test images
            image_path = self.test_images[i % len(self.test_images)]
            
            # Run inference and collect metrics
            run_result = self._run_single(image_path)
            
            # Store results
            results["timings"]["inference_ms"].append(run_result.inference_time_ms)
            results["timings"]["pre_process_ms"].append(run_result.pre_process_time_ms)
            results["timings"]["post_process_ms"].append(run_result.post_process_time_ms)
            results["timings"]["total_ms"].append(run_result.total_time_ms)
            
            if run_result.memory_usage_kb:
                results["memory_usage_kb"].append(run_result.memory_usage_kb)
                
            if run_result.extra_metrics:
                results["extra_metrics"].append(run_result.extra_metrics)
                
        # Calculate statistics
        for metric in results["timings"].keys():
            times = results["timings"][metric]
            results[f"{metric}_stats"] = {
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "p90": float(np.percentile(times, 90))
            }
            
        # Calculate throughput
        avg_total = np.mean(results["timings"]["total_ms"])
        results["throughput_fps"] = 1000.0 / avg_total
        
        if results["memory_usage_kb"]:
            results["avg_memory_mb"] = np.mean(results["memory_usage_kb"]) / 1024
            
        return results

    def _run_single(self, image_path: str):
        """Run single inference with a test image"""
        return self.runner.run_single(image_path)

    def _get_hardware_info(self) -> Dict:
        """Get hardware information"""
        info = {
            "platform": "Unknown",
            "device_name": "Unknown"
        }
        
        # Try to get CPU info
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
        except:
            pass
            
        # Try to get memory info
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        info["total_memory_mb"] = mem_kb / 1024
                        break
        except:
            pass
            
        return info

    def cleanup(self):
        """Clean up resources"""
        if self.runner:
            self.runner.cleanup()
            self.runner = None

def main():
    """Example usage"""
    # Define test configuration
    test_images = [
        "images/cat.jpg",
        "images/dog.jpg",
        "images/person.jpg"
    ]
    
    # Run benchmark for each model
    for model_name, model_path in [
        ("mobilenetv2", "models/mobilenetv2-12.onnx"),
        ("yolov5s", "models/yolov5s.onnx")
    ]:
        print(f"\nRunning benchmark for {model_name}...")
        
        benchmark = TPUBenchmark(
            model_name=model_name,
            model_path=model_path,
            test_images=test_images
        )
        
        try:
            with benchmark:
                results = benchmark.run()
                benchmark.print_results(results)
                if benchmark.config.save_results:
                    benchmark.save_results(results)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")

if __name__ == "__main__":
    main()
