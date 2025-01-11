"""
Base Benchmark Class
Defines the interface for all benchmark implementations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import json
import os
from pathlib import Path
import logging
from datetime import datetime

# Import from runners package
from runners.base import RunnerConfig

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    name: str
    runner_config: RunnerConfig
    warmup_runs: int = 5
    num_runs: int = 100
    output_dir: str = "results"
    save_results: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.runner_config, RunnerConfig):
            raise TypeError("runner_config must be a RunnerConfig instance")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.num_runs < 1:
            raise ValueError("num_runs must be positive")
        if not isinstance(self.output_dir, str):
            raise TypeError("output_dir must be a string")

class BenchmarkBase(ABC):
    """Abstract base class for all benchmarks
    
    This class defines the interface that all benchmark implementations must follow.
    Each benchmark is responsible for orchestrating tests and collecting results
    for a specific type of hardware or operation.
    """
    
    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark with configuration

        Args:
            config (BenchmarkConfig): Configuration for the benchmark

        Raises:
            TypeError: If config is not a BenchmarkConfig instance
        """
        if not isinstance(config, BenchmarkConfig):
            raise TypeError("config must be a BenchmarkConfig instance")
        self.config = config
        self.results: Dict[str, Any] = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the benchmark

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
    def run(self) -> Dict[str, Any]:
        """Run the complete benchmark suite

        Returns:
            Dict[str, Any]: Complete benchmark results

        This method should orchestrate the full benchmark process, including:
        - Running warmup iterations
        - Executing benchmark runs
        - Collecting and processing results
        - Calculating statistics
        """
        pass
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize any required resources

        Returns:
            bool: True if initialization successful, False otherwise

        This method should handle:
        - Setting up hardware resources
        - Loading any required models or data
        - Validating system requirements
        """
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any allocated resources

        This method should ensure proper cleanup of:
        - Hardware resources
        - Memory allocations
        - Temporary files
        - System configurations
        """
        pass
        
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> None:
        """Save benchmark results to file

        Args:
            results (Dict[str, Any]): Results to save
            filename (Optional[str]): Output filename, defaults to benchmark name

        Raises:
            IOError: If unable to save results
        """
        if not self.config.save_results:
            return
            
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.config.name}_benchmark_{timestamp}.json"
                
            output_path = Path(self.config.output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            results_with_metadata = {
                "metadata": {
                    "benchmark_name": self.config.name,
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "name": self.config.name,
                        "warmup_runs": self.config.warmup_runs,
                        "num_runs": self.config.num_runs,
                        "runner_config": self.config.runner_config.__dict__
                    }
                },
                "results": results
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_with_metadata, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            raise IOError(f"Failed to save results: {str(e)}")
        
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a readable format

        Args:
            results (Dict[str, Any]): Results to print
        """
        self.logger.info(f"\n{self.config.name} Benchmark Results")
        self.logger.info("=" * 40)
        
        # Print basic stats
        if "timings" in results:
            self.logger.info("\nTiming Statistics:")
            for metric, stats in results["timings_stats"].items():
                self.logger.info(f"{metric}:")
                for stat, value in stats.items():
                    self.logger.info(f"  {stat}: {value:.2f}")
                    
        # Print throughput if available
        if "throughput_fps" in results:
            self.logger.info(f"\nThroughput: {results['throughput_fps']:.2f} FPS")
            
        # Print memory usage if available
        if "memory" in results and results["memory"]:
            avg_memory_mb = sum(results["memory"]) / len(results["memory"]) / 1024
            self.logger.info(f"Average Memory Usage: {avg_memory_mb:.2f} MB")
            
        # Print any extra metrics
        if "extra_metrics" in results:
            self.logger.info("\nExtra Metrics:")
            for metric, value in results["extra_metrics"].items():
                self.logger.info(f"{metric}: {value}")
                
    def __enter__(self) -> 'BenchmarkBase':
        """Context manager entry

        Returns:
            BenchmarkBase: Self for context management

        Raises:
            RuntimeError: If initialization fails
        """
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize {self.config.name} benchmark")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit

        This method ensures cleanup is performed even if an exception occurs
        """
        self.cleanup()
