"""
Visualization Module
Generates plots and visualizations for benchmark results
"""
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class BenchmarkVisualizer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        
    def create_comparison_plot(self, 
                             results: Dict,
                             output_file: str = "comparison.png"):
        """Create comparison plot of benchmark results"""
        devices = results["devices"]
        metrics = results["metrics"]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Benchmark Comparison")
        
        # Inference Time
        if metrics["inference_time_ms"]:
            times = [t for t in metrics["inference_time_ms"] if t is not None]
            if times:
                ax1.bar(devices[:len(times)], times)
                ax1.set_title("Inference Time")
                ax1.set_ylabel("Time (ms)")
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Throughput
        if metrics["throughput_fps"]:
            fps = [f for f in metrics["throughput_fps"] if f is not None]
            if fps:
                ax2.bar(devices[:len(fps)], fps)
                ax2.set_title("Throughput")
                ax2.set_ylabel("FPS")
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Memory Usage
        if metrics.get("memory_usage_kb"):
            memory = [m/1024 for m in metrics["memory_usage_kb"] if m is not None]
            if memory:
                ax3.bar(devices[:len(memory)], memory)
                ax3.set_title("Memory Usage")
                ax3.set_ylabel("MB")
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Normalized Performance Score
        if metrics["throughput_fps"] and metrics["inference_time_ms"]:
            # Calculate normalized score (higher is better)
            scores = []
            for fps, time in zip(metrics["throughput_fps"], 
                               metrics["inference_time_ms"]):
                if fps is not None and time is not None:
                    score = (fps / max(metrics["throughput_fps"])) * 0.6 + \
                           (min(metrics["inference_time_ms"]) / time) * 0.4
                    scores.append(score * 100)  # Convert to percentage
                else:
                    scores.append(None)
                    
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                ax4.bar(devices[:len(valid_scores)], valid_scores)
                ax4.set_title("Performance Score")
                ax4.set_ylabel("Score")
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file)
        plt.close()
        
    def create_timeline_plot(self,
                           results: List[Dict],
                           metric: str,
                           output_file: str = "timeline.png"):
        """Create timeline plot for a specific metric"""
        dates = [r["timestamp"] for r in results]
        values = [r["performance"][metric] for r in results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, marker='o')
        plt.title(f"{metric} Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file)
        plt.close()
        
    def create_detail_plots(self,
                          result: Dict,
                          output_prefix: str):
        """Create detailed plots for a single benchmark result"""
        # Memory Bandwidth Plot (if available)
        if "memory_benchmark" in result:
            mem_data = result["memory_benchmark"]
            plt.figure(figsize=(10, 6))
            plt.plot(mem_data["array_sizes"], mem_data["read_bandwidth"],
                    label="Read")
            plt.plot(mem_data["array_sizes"], mem_data["write_bandwidth"],
                    label="Write")
            plt.xlabel("Array Size (MB)")
            plt.ylabel("Bandwidth (MB/s)")
            plt.title("Memory Bandwidth")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{output_prefix}_memory.png")
            plt.close()
            
        # Inference Distribution Plot
        if "timings" in result:
            plt.figure(figsize=(10, 6))
            plt.hist(result["timings"]["inference_ms"], bins=30)
            plt.xlabel("Inference Time (ms)")
            plt.ylabel("Count")
            plt.title("Inference Time Distribution")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{output_prefix}_inference_dist.png")
            plt.close()
            
    def create_report_plots(self, results: Dict):
        """Create all plots for a benchmark report"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Basic comparison plots
        self.create_comparison_plot(results)
        
        # Detail plots for each device
        for device, result in results.items():
            self.create_detail_plots(result, f"{device}_details")
            
    def create_heatmap(self,
                      data: np.ndarray,
                      labels: List[str],
                      title: str,
                      output_file: str):
        """Create heatmap visualization"""
        plt.figure(figsize=(10, 8))
        plt.imshow(data)
        plt.colorbar()
        
        # Add labels
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        
        # Add title
        plt.title(title)
        
        # Add value annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file)
        plt.close()

def main():
    """Example usage"""
    visualizer = BenchmarkVisualizer()
    
    # Example comparison data
    example_results = {
        "devices": ["Device A", "Device B", "Device C"],
        "metrics": {
            "inference_time_ms": [10.0, 15.0, 20.0],
            "throughput_fps": [100.0, 66.7, 50.0],
            "memory_usage_kb": [1024, 2048, 4096]
        }
    }
    
    visualizer.create_comparison_plot(example_results)
    print("Example plots generated in results directory")

if __name__ == "__main__":
    main()