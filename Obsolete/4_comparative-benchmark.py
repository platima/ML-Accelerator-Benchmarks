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
import matplotlib.pyplot as plt
from tabulate import tabulate

@dataclass
class ComparativeBenchmarkResult:
    """Results that can be fairly compared across platforms"""
    model_name: str
    platform: str           # rv1106, cv180x, etc.
    npu_type: str          # rknn or cvitek
    inference_time: float   # in milliseconds
    throughput: float      # inferences/second
    memory_usage: float    # in MB
    power_usage: float     # in watts
    input_shape: tuple
    batch_size: int
    quantization: str      # i8/INT8 or fp/BF16

class ComparativeAnalyzer:
    def __init__(self, results_files: List[str]):
        """
        Initialize analyzer with multiple benchmark result files
        for comparison across platforms
        """
        self.results = []
        self.logger = self._setup_logging()
        
        for result_file in results_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                for result in data['results']:
                    self.results.append(ComparativeBenchmarkResult(
                        model_name=result['model_name'],
                        platform=data['platform_info']['platform'],
                        npu_type=data['platform_info']['npu_type'],
                        inference_time=result['inference_time'],
                        throughput=result['throughput'],
                        memory_usage=result['memory_usage'],
                        power_usage=result.get('power_usage', 0),
                        input_shape=tuple(result['input_shape']),
                        batch_size=result['input_shape'][0],
                        quantization=result['quantization']
                    ))

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("ComparativeAnalyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def generate_comparison_table(self) -> str:
        """Generate a comparison table of results"""
        headers = ['Model', 'Platform', 'NPU', 'Inference(ms)', 'FPS', 'Memory(MB)', 'Power(W)']
        rows = []
        
        for result in self.results:
            rows.append([
                result.model_name,
                result.platform,
                result.npu_type,
                f"{result.inference_time:.2f}",
                f"{result.throughput:.2f}",
                f"{result.memory_usage:.1f}",
                f"{result.power_usage:.3f}" if result.power_usage else "N/A"
            ])
        
        return tabulate(rows, headers=headers, tablefmt="grid")

    def plot_performance_comparison(self, output_path: str):
        """Generate performance comparison plots"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Inference Time Comparison
        plt.subplot(2, 2, 1)
        platforms = [r.platform for r in self.results]
        times = [r.inference_time for r in self.results]
        plt.bar(platforms, times)
        plt.title('Inference Time Comparison')
        plt.ylabel('Time (ms)')
        
        # Subplot 2: Throughput Comparison
        plt.subplot(2, 2, 2)
        throughputs = [r.throughput for r in self.results]
        plt.bar(platforms, throughputs)
        plt.title('Throughput Comparison')
        plt.ylabel('FPS')
        
        # Subplot 3: Memory Usage
        plt.subplot(2, 2, 3)
        memory = [r.memory_usage for r in self.results]
        plt.bar(platforms, memory)
        plt.title('Memory Usage Comparison')
        plt.ylabel('Memory (MB)')
        
        # Subplot 4: Power Usage (if available)
        plt.subplot(2, 2, 4)
        power = [r.power_usage for r in self.results if r.power_usage]
        if power:
            power_platforms = [r.platform for r in self.results if r.power_usage]
            plt.bar(power_platforms, power)
            plt.title('Power Usage Comparison')
            plt.ylabel('Power (W)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def generate_efficiency_metrics(self) -> Dict:
        """
        Calculate efficiency metrics that can be fairly compared:
        - Inferences per Watt
        - Inferences per MB
        - Energy cost per inference
        """
        metrics = {}
        
        for result in self.results:
            platform_metrics = {
                'inferences_per_watt': (
                    result.throughput / result.power_usage 
                    if result.power_usage else None
                ),
                'inferences_per_mb': result.throughput / result.memory_usage,
                'energy_per_inference': (
                    (result.power_usage / result.throughput) 
                    if result.power_usage else None
                )
            }
            metrics[f"{result.platform}_{result.model_name}"] = platform_metrics
            
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Compare NPU benchmark results')
    parser.add_argument('--results', nargs='+', required=True,
                       help='Paths to benchmark result JSON files')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for comparison results')
    args = parser.parse_args()

    analyzer = ComparativeAnalyzer(args.results)
    
    # Generate comparison table
    comparison_table = analyzer.generate_comparison_table()
    print("\nPerformance Comparison Table:")
    print(comparison_table)
    
    # Save comparison table
    with open(os.path.join(args.output_dir, 'comparison_table.txt'), 'w') as f:
        f.write(comparison_table)
    
    # Generate and save plots
    analyzer.plot_performance_comparison(
        os.path.join(args.output_dir, 'performance_comparison.png')
    )
    
    # Calculate efficiency metrics
    metrics = analyzer.generate_efficiency_metrics()
    print("\nEfficiency Metrics:")
    print(json.dumps(metrics, indent=2))
    
    with open(os.path.join(args.output_dir, 'efficiency_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
