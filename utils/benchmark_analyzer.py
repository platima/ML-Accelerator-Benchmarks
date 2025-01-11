#!/usr/bin/env python3
"""
SoC Benchmark Analyzer
Analyzes and compares benchmark results from various SoCs.
Features:
- Performance scoring
- Comparative analysis
- Visualization
- Detailed reporting
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class BenchmarkResult:
    soc_name: str
    model_name: str
    accelerator_type: str
    inference_time: float
    memory_usage: float
    power_usage: Optional[float]
    throughput: float
    quantization: Optional[str]
    input_shape: tuple
    cpu_usage: float
    temperature: Optional[float]
    timestamp: str
    batch_size: int

class PerformanceScorer:
    def __init__(self):
        # Reference values for normalization (based on typical edge NPU performance)
        self.reference = {
            "inference_time": 10.0,    # ms
            "power_usage": 2.0,        # watts
            "memory_usage": 100.0,     # MB
            "temperature": 60.0,       # celsius
            "throughput": 100.0,       # FPS
            "cpu_usage": 20.0          # percent
        }

        # Scoring weights (total = 100)
        self.weights = {
            "inference": 40,   # Speed is most important
            "power": 20,       # Power efficiency
            "memory": 15,      # Memory usage
            "thermal": 10,     # Temperature management
            "cpu": 10,         # CPU utilization
            "batch": 5         # Batch processing capability
        }

    def calculate_score(self, result: BenchmarkResult) -> Dict:
        scores = {}
        
        # Inference Score (lower is better)
        inference_ratio = self.reference["inference_time"] / max(result.inference_time, 0.1)
        scores["inference"] = min(inference_ratio * self.weights["inference"], self.weights["inference"])
        
        # Power Score (lower is better)
        if result.power_usage:
            power_ratio = self.reference["power_usage"] / max(result.power_usage, 0.1)
            scores["power"] = min(power_ratio * self.weights["power"], self.weights["power"])
        else:
            scores["power"] = 0
        
        # Memory Score (lower is better)
        memory_ratio = self.reference["memory_usage"] / max(result.memory_usage, 1.0)
        scores["memory"] = min(memory_ratio * self.weights["memory"], self.weights["memory"])
        
        # Thermal Score (lower is better)
        if result.temperature:
            temp_ratio = self.reference["temperature"] / max(result.temperature, 30.0)
            scores["thermal"] = min(temp_ratio * self.weights["thermal"], self.weights["thermal"])
        else:
            scores["thermal"] = 0
        
        # CPU Usage Score (lower is better)
        cpu_ratio = self.reference["cpu_usage"] / max(result.cpu_usage, 1.0)
        scores["cpu"] = min(cpu_ratio * self.weights["cpu"], self.weights["cpu"])
        
        # Batch Processing Score (higher is better)
        batch_ratio = result.throughput / self.reference["throughput"]
        scores["batch"] = min(batch_ratio * self.weights["batch"], self.weights["batch"])
        
        # Calculate total score (0-100)
        total_score = sum(scores.values())
        
        # Calculate performance metrics
        metrics = {
            "inferences_per_watt": (
                result.throughput / result.power_usage if result.power_usage else None
            ),
            "inferences_per_mb": result.throughput / result.memory_usage,
            "power_efficiency": (
                result.throughput / (result.power_usage * result.memory_usage) 
                if result.power_usage else None
            )
        }
        
        return {
            "total_score": round(total_score, 1),
            "subscores": {k: round(v, 1) for k, v in scores.items()},
            "metrics": {k: round(v, 2) if v else None for k, v in metrics.items()}
        }

    def get_performance_tier(self, score: float) -> str:
        """Maps score to a performance tier"""
        if score >= 90:
            return "Elite (Server Grade)"
        elif score >= 80:
            return "Premium (Edge Server)"
        elif score >= 70:
            return "High-End (Edge Device)"
        elif score >= 60:
            return "Mid-Range (Mobile)"
        elif score >= 50:
            return "Entry-Level (IoT)"
        elif score >= 40:
            return "Basic (MCU)"
        else:
            return "Limited (CPU Only)"

class BenchmarkAnalyzer:
    def __init__(self, results_files: List[str]):
        self.results = []
        self.scorer = PerformanceScorer()
        
        for file_path in results_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract result from nested structure
                result_data = data['results']
                # Convert input_shape from list to tuple
                result_data['input_shape'] = tuple(result_data['input_shape'])
                self.results.append(BenchmarkResult(**result_data))

    def generate_comparison_table(self) -> str:
        """Generates a formatted comparison table"""
        headers = ["SoC", "Score", "Tier", "FPS", "Power(W)", "Temp(°C)"]
        rows = []
        
        for result in sorted(self.results, 
                           key=lambda x: self.scorer.calculate_score(x)["total_score"],
                           reverse=True):
            score_data = self.scorer.calculate_score(result)
            rows.append([
                result.soc_name,
                f"{score_data['total_score']:>5.1f}",
                self.scorer.get_performance_tier(score_data['total_score']),
                f"{result.throughput:>6.1f}",
                f"{result.power_usage:>5.1f}" if result.power_usage else "N/A",
                f"{result.temperature:>5.1f}" if result.temperature else "N/A"
            ])
        
        # Format as table
        col_widths = [
            max(len(str(row[i])) for row in rows + [headers])
            for i in range(len(headers))
        ]
        
        table = ""
        # Add headers
        for i, header in enumerate(headers):
            table += f"{header:<{col_widths[i]}} "
        table += "\n" + "-" * sum(col_widths) + "\n"
        
        # Add rows
        for row in rows:
            for i, item in enumerate(row):
                table += f"{str(item):<{col_widths[i]}} "
            table += "\n"
        
        return table

    def generate_visualizations(self, output_path: str):
        """Generates comparative visualizations"""
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("SoC Benchmark Comparison")
        
        # Performance Scores
        ax1 = plt.subplot(2, 2, 1)
        scores = [self.scorer.calculate_score(r)["total_score"] for r in self.results]
        names = [r.soc_name for r in self.results]
        ax1.bar(names, scores)
        ax1.set_title("Performance Scores")
        ax1.set_ylabel("Score")
        plt.xticks(rotation=45)
        
        # Inference Time
        ax2 = plt.subplot(2, 2, 2)
        times = [r.inference_time for r in self.results]
        ax2.bar(names, times)
        ax2.set_title("Inference Time")
        ax2.set_ylabel("Time (ms)")
        plt.xticks(rotation=45)
        
        # Power Usage
        ax3 = plt.subplot(2, 2, 3)
        power = [r.power_usage for r in self.results if r.power_usage]
        power_names = [r.soc_name for r in self.results if r.power_usage]
        ax3.bar(power_names, power)
        ax3.set_title("Power Usage")
        ax3.set_ylabel("Power (W)")
        plt.xticks(rotation=45)
        
        # Throughput
        ax4 = plt.subplot(2, 2, 4)
        fps = [r.throughput for r in self.results]
        ax4.bar(names, fps)
        ax4.set_title("Throughput")
        ax4.set_ylabel("FPS")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def generate_report(self, output_path: str):
        """Generates a comprehensive report"""
        report = "SoC Benchmark Analysis Report\n"
        report += "=" * 80 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add comparison table
        report += "Performance Comparison\n"
        report += "-" * 80 + "\n"
        report += self.generate_comparison_table()
        report += "\n"
        
        # Detailed results for each SoC
        report += "Detailed Analysis\n"
        report += "-" * 80 + "\n"
        
        for result in sorted(self.results, 
                           key=lambda x: self.scorer.calculate_score(x)["total_score"],
                           reverse=True):
            score_data = self.scorer.calculate_score(result)
            
            report += f"\nSoC: {result.soc_name}\n"
            report += f"Model: {result.model_name}\n"
            report += f"Accelerator: {result.accelerator_type}\n"
            report += f"Overall Score: {score_data['total_score']:.1f} "
            report += f"({self.scorer.get_performance_tier(score_data['total_score'])})\n"
            
            report += "\nPerformance Subscores:\n"
            for name, score in score_data['subscores'].items():
                report += f"  {name:<12}: {score:>6.1f}\n"
            
            report += "\nPerformance Metrics:\n"
            report += f"  Inference Time: {result.inference_time:>6.1f} ms\n"
            report += f"  Throughput: {result.throughput:>6.1f} FPS\n"
            report += f"  Memory Usage: {result.memory_usage:>6.1f} MB\n"
            if result.power_usage:
                report += f"  Power Usage: {result.power_usage:>6.1f} W\n"
            if result.temperature:
                report += f"  Temperature: {result.temperature:>6.1f}°C\n"
            
            report += "\nEfficiency Metrics:\n"
            for name, value in score_data['metrics'].items():
                if value is not None:
                    report += f"  {name:<20}: {value:>6.2f}\n"
            
            report += "-" * 40 + "\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Analyze SoC benchmark results')
    parser.add_argument('--results', nargs='+', required=True,
                       help='Paths to benchmark result JSON files')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for analysis files')
    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.results)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'benchmark_report.txt')
    analyzer.generate_report(report_path)
    print(f"Report generated: {report_path}")
    
    # Generate visualizations
    viz_path = os.path.join(args.output_dir, 'benchmark_comparison.png')
    analyzer.generate_visualizations(viz_path)
    print(f"Visualizations generated: {viz_path}")

if __name__ == '__main__':
    main()
