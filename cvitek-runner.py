"""
CVITEK TPU-MLIR benchmark runner
Supports CV1800B/SG2002/SG2000
"""
import numpy as np
import time
from typing import Dict, Optional
from dataclasses import dataclass
import cvi_toolkit
from cvi_toolkit import Model, Tensor

@dataclass
class CVITEKConfig:
    chip: str  # 'cv1800b'/'sg2002'/'sg2000'
    quantization: bool = True
    opt_level: int = 3
    batch_size: int = 1

class CVITEKBenchmark:
    def __init__(self, model_path: str, config: CVITEKConfig):
        self.model_path = model_path
        self.config = config
        self.model = None
        
    def prepare_model(self) -> bool:
        """Initialize and load CVITEK model"""
        try:
            # Create model instance
            self.model = Model(self.config.chip)
            
            # Load model
            self.model.load(self.model_path)
            
            # Apply quantization if requested
            if self.config.quantization:
                self.model.quantize(
                    calibration_method='minmax',
                    optimization_level=self.config.opt_level
                )
            
            # Compile model
            self.model.compile()
            
            return True
            
        except Exception as e:
            print(f"Error preparing model: {str(e)}")
            return False
            
    def run_benchmark(self, 
                     input_shape: tuple,
                     num_warmup: int = 5,
                     num_runs: int = 100) -> Dict:
        """Run inference benchmark"""
        try:
            # Create input tensor
            input_data = np.random.rand(*input_shape).astype(np.float32)
            input_tensor = Tensor.from_numpy(input_data)
            
            # Warmup runs
            print("Performing warmup runs...")
            for _ in range(num_warmup):
                self.model.inference(inputs=[input_tensor])
            
            # Benchmark runs
            print(f"Running {num_runs} iterations...")
            times = []
            
            for i in range(num_runs):
                if i % 10 == 0:
                    print(f"Progress: {i}/{num_runs}")
                    
                start = time.perf_counter()
                outputs = self.model.inference(inputs=[input_tensor])
                times.append((time.perf_counter() - start) * 1000)  # Convert to ms
                
            # Calculate statistics
            times = np.array(times)
            results = {
                "chip": self.config.chip,
                "quantization": self.config.quantization,
                "input_shape": input_shape,
                "avg_inference_ms": float(np.mean(times)),
                "min_inference_ms": float(np.min(times)),
                "max_inference_ms": float(np.max(times)),
                "std_inference_ms": float(np.std(times)),
                "throughput_fps": float(1000 / np.mean(times))
            }
            
            return results
            
        except Exception as e:
            print(f"Error during benchmark: {str(e)}")
            return None
            
    def cleanup(self):
        """Clean up CVITEK runtime"""
        if self.model is not None:
            self.model.release()
            
def main():
    # Example usage for YOLOv5
    config = CVITEKConfig(
        chip='cv1800b',
        quantization=True
    )
    
    benchmark = CVITEKBenchmark(
        model_path='yolov5s.onnx',
        config=config
    )
    
    try:
        if benchmark.prepare_model():
            results = benchmark.run_benchmark(
                input_shape=(1, 3, 640, 640)  # YOLOv5 default input size
            )
            if results:
                print("\nBenchmark Results:")
                for k, v in results.items():
                    print(f"{k}: {v}")
    finally:
        benchmark.cleanup()

if __name__ == '__main__':
    main()
