"""
base_runner.py
Base Runner Class with improved type hints, validation, and error handling
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np
import time
import logging
from enum import Enum

class QuantizationType(Enum):
    """Supported quantization types"""
    INT8 = "int8"
    BF16 = "bf16"
    NONE = None

@dataclass
class RunnerConfig:
    """Configuration for benchmark runners"""
    input_shape: tuple
    batch_size: int = 1
    num_warmup: int = 10
    num_runs: int = 100
    quantization: Optional[QuantizationType] = None
    threads: int = 1
    model_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        if not isinstance(self.input_shape, tuple):
            raise TypeError("input_shape must be a tuple")
        if not all(isinstance(dim, int) and dim > 0 for dim in self.input_shape):
            raise ValueError("input_shape must contain positive integers")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_warmup < 0:
            raise ValueError("num_warmup cannot be negative")
        if self.num_runs < 1:
            raise ValueError("num_runs must be positive")
        if self.threads < 1:
            raise ValueError("threads must be positive")

@dataclass
class RunnerResult:
    """Standard result format for all runners"""
    inference_time_ms: float
    pre_process_time_ms: float
    post_process_time_ms: float
    total_time_ms: float
    memory_usage_kb: Optional[int] = None
    accuracy: Optional[float] = None
    output: Optional[np.ndarray] = None
    extra_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate result data"""
        if self.inference_time_ms < 0:
            raise ValueError("inference_time_ms cannot be negative")
        if self.pre_process_time_ms < 0:
            raise ValueError("pre_process_time_ms cannot be negative")
        if self.post_process_time_ms < 0:
            raise ValueError("post_process_time_ms cannot be negative")
        if self.memory_usage_kb is not None and self.memory_usage_kb < 0:
            raise ValueError("memory_usage_kb cannot be negative")
        if self.accuracy is not None and not 0 <= self.accuracy <= 1:
            raise ValueError("accuracy must be between 0 and 1")

class BenchmarkRunner(ABC):
    """Abstract base class for all benchmark runners"""
    
    def __init__(self, config: RunnerConfig) -> None:
        """Initialize the runner with configuration

        Args:
            config (RunnerConfig): Configuration parameters for the runner

        Raises:
            TypeError: If config is not a RunnerConfig instance
        """
        if not isinstance(config, RunnerConfig):
            raise TypeError("config must be a RunnerConfig instance")
        self.config = config
        self._initialized = False
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the runner

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the runner and required resources

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    def run_single(self, input_data: Optional[np.ndarray] = None) -> RunnerResult:
        """Run a single inference/computation

        Args:
            input_data (Optional[np.ndarray]): Input data for computation

        Returns:
            RunnerResult: Results from single run

        Raises:
            RuntimeError: If runner not initialized
        """
        pass
        
    def run_benchmark(self, input_data: Optional[np.ndarray] = None) -> Dict:
        """Run complete benchmark suite

        Args:
            input_data (Optional[np.ndarray]): Input data for benchmark

        Returns:
            Dict: Complete benchmark results

        Raises:
            RuntimeError: If initialization fails
        """
        if not self._initialized:
            self.logger.info("Initializing runner...")
            if not self.initialize():
                raise RuntimeError("Failed to initialize runner")
            self._initialized = True
            
        results = {
            "config": self.config.__dict__,
            "timings": {
                "inference_ms": [],
                "pre_process_ms": [],
                "post_process_ms": [],
                "total_ms": []
            },
            "memory": [],
            "accuracy": [],
            "extra_metrics": []
        }
        
        try:
            # Warmup runs
            self.logger.info(f"Performing {self.config.num_warmup} warmup runs...")
            for _ in range(self.config.num_warmup):
                self.run_single(input_data)
            
            # Benchmark runs
            self.logger.info(f"Performing {self.config.num_runs} benchmark runs...")
            for i in range(self.config.num_runs):
                if i % 10 == 0:
                    self.logger.info(f"Progress: {i}/{self.config.num_runs}")
                    
                result = self.run_single(input_data)
                
                # Store results
                results["timings"]["inference_ms"].append(result.inference_time_ms)
                results["timings"]["pre_process_ms"].append(result.pre_process_time_ms)
                results["timings"]["post_process_ms"].append(result.post_process_time_ms)
                results["timings"]["total_ms"].append(result.total_time_ms)
                
                if result.memory_usage_kb is not None:
                    results["memory"].append(result.memory_usage_kb)
                if result.accuracy is not None:
                    results["accuracy"].append(result.accuracy)
                    
                # Store any extra metrics
                if result.extra_metrics:
                    results["extra_metrics"].append(result.extra_metrics)
                    
            # Calculate statistics
            for metric, values in results["timings"].items():
                results[f"{metric}_stats"] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p90": float(np.percentile(values, 90))
                }
                
            # Calculate throughput
            avg_total = np.mean(results["timings"]["total_ms"])
            results["throughput_fps"] = 1000.0 / avg_total

        except Exception as e:
            self.logger.error(f"Error during benchmark: {str(e)}")
            raise
            
        return results
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources"""
        pass
        
    def __enter__(self) -> 'BenchmarkRunner':
        """Context manager entry

        Returns:
            BenchmarkRunner: Self for context management
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.cleanup()
