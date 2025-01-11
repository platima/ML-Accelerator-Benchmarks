"""
Simplified visualization module for MicroPython
Provides basic text-based visualizations and data formatting
"""

class BasicVisualizer:
    """Simplified visualizer for MicroPython environments"""
    def __init__(self):
        self.max_width = 40  # Maximum width for ASCII charts
        
    def create_text_bar(self, value, max_value, width=None):
        """Create a simple ASCII bar chart"""
        width = width or self.max_width
        fill = int((value / max_value) * width)
        return "[" + "#" * fill + "-" * (width - fill) + "]"
        
    def format_number(self, num):
        """Format numbers with appropriate units"""
        if num >= 1e9:
            return f"{num/1e9:.1f}G"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.1f}"
            
    def show_benchmark_results(self, results):
        """Display benchmark results in text format"""
        print("\nBenchmark Results")
        print("=" * self.max_width)
        
        for benchmark, data in results.items():
            print(f"\n{benchmark.upper()} Benchmark:")
            print("-" * self.max_width)
            
            if "performance" in data:
                perf = data["performance"]
                if "throughput_fps" in perf:
                    fps = perf["throughput_fps"]
                    max_fps = 100  # Adjust based on expected range
                    print(f"Throughput: {fps:.1f} FPS")
                    print(self.create_text_bar(fps, max_fps))
                    
                if "inference_time_ms" in perf:
                    time_ms = perf["inference_time_ms"]
                    print(f"Inference: {time_ms:.1f} ms")
                    print(self.create_text_bar(1/time_ms, 1))
                    
                if "memory_usage_kb" in perf:
                    mem = perf["memory_usage_kb"]
                    print(f"Memory: {self.format_number(mem)}B")
                    
            if "memory_benchmark" in data:
                mem = data["memory_benchmark"]
                if "read_bandwidth" in mem and "write_bandwidth" in mem:
                    print("\nMemory Bandwidth:")
                    read = max(mem["read_bandwidth"])
                    write = max(mem["write_bandwidth"])
                    print(f"Read:  {self.format_number(read)}B/s")
                    print(self.create_text_bar(read, max(read, write)))
                    print(f"Write: {self.format_number(write)}B/s")
                    print(self.create_text_bar(write, max(read, write)))
                    
    def show_comparison(self, results1, results2, name1="Before", name2="After"):
        """Compare two benchmark results"""
        print(f"\nComparison: {name1} vs {name2}")
        print("=" * self.max_width)
        
        metrics = ["throughput_fps", "inference_time_ms", "memory_usage_kb"]
        
        for metric in metrics:
            if metric in results1.get("performance", {}) and \
               metric in results2.get("performance", {}):
                val1 = results1["performance"][metric]
                val2 = results2["performance"][metric]
                diff = ((val2 - val1) / val1) * 100
                
                print(f"\n{metric}:")
                print(f"{name1}: {self.format_number(val1)}")
                print(f"{name2}: {self.format_number(val2)}")
                print(f"Change: {diff:+.1f}%")
                
    def create_summary(self, results):
        """Create a compact results summary"""
        summary = []
        
        for benchmark, data in results.items():
            if "performance" in data:
                perf = data["performance"]
                metrics = []
                
                if "throughput_fps" in perf:
                    metrics.append(f"{perf['throughput_fps']:.1f}fps")
                if "inference_time_ms" in perf:
                    metrics.append(f"{perf['inference_time_ms']:.1f}ms")
                if "memory_usage_kb" in perf:
                    metrics.append(f"{self.format_number(perf['memory_usage_kb'])}B")
                    
                summary.append(f"{benchmark}: {', '.join(metrics)}")
                
        return "\n".join(summary)

# Export for conditional import
BASIC_VISUALIZER = BasicVisualizer