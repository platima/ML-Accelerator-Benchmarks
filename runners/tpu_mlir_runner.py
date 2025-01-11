"""
tpu_mlir_runner.py
TPU/NPU Runner for RKNN and CVITEK platforms with improved error handling and type hints
"""
from typing import Dict, Optional, Union, Any
import os
import time
import numpy as np
import cv2
from pathlib import Path

from .base import (
    BenchmarkRunner, 
    RunnerConfig, 
    RunnerResult,
    QuantizationType
)

class AcceleratorWrapper:
    """Wrapper for different accelerator backends"""
    def __init__(self, accelerator_type: str, instance: Any):
        self.type = accelerator_type
        self.instance = instance

    def cleanup(self) -> None:
        """Clean up accelerator resources"""
        if self.type == "RKNN":
            self.instance.release()
        elif self.type == "CVITEK":
            self.instance.release()

class TPURunner(BenchmarkRunner):
    """AI Accelerator runner for RKNN and CVITEK platforms"""
    
    def __init__(self, config: Optional[RunnerConfig] = None) -> None:
        if config is None:
            # Default to MobileNetV2 config
            config = RunnerConfig(
                input_shape=(1, 3, 224, 224),
                batch_size=1,
                num_warmup=5,
                num_runs=100,
                quantization=QuantizationType.INT8,
                model_path=str(Path("models/mobilenetv2-12.onnx"))
            )
        super().__init__(config)
        self.accelerator: Optional[AcceleratorWrapper] = None
        self._detect_accelerator()
        
    def _detect_accelerator(self) -> None:
        """Auto-detect available accelerator

        Raises:
            RuntimeError: If no supported accelerator is found
        """
        # Try RKNN
        try:
            from rknn.api import RKNN
            self.accelerator = AcceleratorWrapper(
                accelerator_type="RKNN",
                instance=RKNN()
            )
            self.logger.info("Using RKNN accelerator")
            return
        except ImportError:
            self.logger.debug("RKNN not available")
            
        # Try CVITEK
        try:
            import cvi_toolkit
            from cvi_toolkit import Model
            self.accelerator = AcceleratorWrapper(
                accelerator_type="CVITEK",
                instance=Model()
            )
            self.logger.info("Using CVITEK accelerator")
            return
        except ImportError:
            self.logger.debug("CVITEK not available")
            
        raise RuntimeError("No supported AI accelerator found")

    def initialize(self) -> bool:
        """Initialize accelerator and load model

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            self.logger.error(f"Model not found: {self.config.model_path}")
            return False
            
        try:
            if self.accelerator.type == "RKNN":
                return self._init_rknn()
            elif self.accelerator.type == "CVITEK":
                return self._init_cvitek()
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize accelerator: {str(e)}")
            return False

    def _init_rknn(self) -> bool:
        """Initialize RKNN accelerator

        Returns:
            bool: True if initialization successful
        """
        rknn = self.accelerator.instance
        
        # Load and initialize model
        ret = rknn.load_rknn(self.config.model_path)
        if ret != 0:
            self.logger.error("Failed to load RKNN model")
            return False
            
        ret = rknn.init_runtime()
        if ret != 0:
            self.logger.error("Failed to initialize RKNN runtime")
            return False
            
        return True

    def _init_cvitek(self) -> bool:
        """Initialize CVITEK accelerator

        Returns:
            bool: True if initialization successful
        """
        model = self.accelerator.instance
        
        try:
            # Load model
            model.load(self.config.model_path)
            
            # Set quantization
            if self.config.quantization == QuantizationType.INT8:
                model.quantize(
                    calibration_method='minmax',
                    quantization_type='asymmetric'
                )
            elif self.config.quantization == QuantizationType.BF16:
                model.quantize(
                    calibration_method='minmax',
                    quantization_type='bf16'
                )
                
            # Compile model
            model.compile()
            return True
        except Exception as e:
            self.logger.error(f"CVITEK initialization failed: {str(e)}")
            return False

    def _preprocess_input(self, input_data: Optional[np.ndarray]) -> np.ndarray:
        """Preprocess input data according to model requirements

        Args:
            input_data (Optional[np.ndarray]): Input data to preprocess

        Returns:
            np.ndarray: Preprocessed input data
        """
        if input_data is None:
            # Create dummy input if none provided
            return np.random.rand(*self.config.input_shape).astype(np.float32)
            
        # Resize if needed
        if input_data.shape[-2:] != self.config.input_shape[-2:]:
            input_data = cv2.resize(
                input_data, 
                self.config.input_shape[-2:]
            )
            
        # Add batch dimension if needed
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, 0)
            
        return input_data.astype(np.float32)

    def run_single(self, input_data: Optional[np.ndarray] = None) -> RunnerResult:
        """Run single inference

        Args:
            input_data (Optional[np.ndarray]): Input data for inference

        Returns:
            RunnerResult: Results from single inference

        Raises:
            RuntimeError: If accelerator not initialized
        """
        if not self.accelerator:
            raise RuntimeError("Accelerator not initialized")

        pre_start = time.perf_counter()
        processed_input = self._preprocess_input(input_data)
        pre_time = time.perf_counter() - pre_start

        infer_start = time.perf_counter()
        if self.accelerator.type == "RKNN":
            outputs = self.accelerator.instance.inference(
                inputs=[processed_input]
            )
        else:  # CVITEK
            outputs = self.accelerator.instance.inference(
                inputs=[processed_input]
            )
        infer_time = time.perf_counter() - infer_start

        post_start = time.perf_counter()
        # Get memory usage if available
        try:
            import psutil
            memory_kb = psutil.Process().memory_info().rss // 1024
        except:
            memory_kb = None
        post_time = time.perf_counter() - post_start

        return RunnerResult(
            inference_time_ms=infer_time * 1000,
            pre_process_time_ms=pre_time * 1000,
            post_process_time_ms=post_time * 1000,
            total_time_ms=(infer_time + pre_time + post_time) * 1000,
            memory_usage_kb=memory_kb,
            output=outputs[0] if outputs else None,
            extra_metrics={
                "accelerator_type": self.accelerator.type,
                "quantization": self.config.quantization.value
                if self.config.quantization else None
            }
        )

    def cleanup(self) -> None:
        """Clean up accelerator resources"""
        if self.accelerator:
            self.accelerator.cleanup()
            self.accelerator = None
            self.logger.info("Accelerator resources cleaned up")

def main() -> None:
    """Example usage"""
    # MobileNetV2 configuration
    config = RunnerConfig(
        input_shape=(1, 3, 224, 224),
        model_path=str(Path("models/mobilenetv2-12.onnx")),
        quantization=QuantizationType.INT8,
        num_warmup=5,
        num_runs=100
    )
    
    with TPURunner(config) as runner:
        try:
            results = runner.run_benchmark()
            print("\nTPU Benchmark Results:")
            print(f"Accelerator: {results['extra_metrics'][0]['accelerator_type']}")
            print(f"Average Inference Time: {results['timings_stats']['inference_ms']['mean']:.2f} ms")
            print(f"Throughput: {results['throughput_fps']:.2f} FPS")
            if results.get('memory'):
                print(f"Memory Usage: {results['memory'][0] / 1024:.2f} MB")
        except Exception as e:
            print(f"Benchmark failed: {str(e)}")

if __name__ == "__main__":
    main()
