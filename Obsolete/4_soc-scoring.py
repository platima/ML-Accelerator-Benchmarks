class PerformanceScorer:
    """
    Calculates a weighted score (0-1000) based on:
    - Raw inference speed (40%)
    - Energy efficiency (20%)
    - Memory efficiency (20%)
    - Temperature stability (10%)
    - Batch processing capability (10%)
    """
    
    def __init__(self, base_model="mobilenet_v2"):
        self.base_model = base_model
        # Reference values for normalization (based on typical CV1800B performance)
        self.reference = {
            "inference_time": 10.0,    # ms
            "power_usage": 2.0,        # watts
            "memory_usage": 100.0,     # MB
            "temperature": 60.0,       # celsius
            "batch_throughput": 100.0  # FPS for batch size 8
        }

    def calculate_score(self, result: BenchmarkResult) -> Dict[str, float]:
        # Initialize subscores
        scores = {
            "inference": 0.0,
            "energy": 0.0,
            "memory": 0.0,
            "thermal": 0.0,
            "batch": 0.0
        }
        
        # Inference Speed Score (40%)
        # Lower is better, normalized against reference
        inference_ratio = self.reference["inference_time"] / max(result.inference_time, 0.1)
        scores["inference"] = min(inference_ratio * 400, 400)
        
        # Energy Efficiency Score (20%)
        # Lower is better
        if result.power_usage:
            power_ratio = self.reference["power_usage"] / max(result.power_usage, 0.1)
            scores["energy"] = min(power_ratio * 200, 200)
        
        # Memory Efficiency Score (20%)
        # Lower is better
        memory_ratio = self.reference["memory_usage"] / max(result.memory_usage, 1.0)
        scores["memory"] = min(memory_ratio * 200, 200)
        
        # Thermal Efficiency Score (10%)
        # Lower is better
        if result.temperature:
            temp_ratio = self.reference["temperature"] / max(result.temperature, 30.0)
            scores["thermal"] = min(temp_ratio * 100, 100)
        
        # Batch Processing Score (10%)
        # Higher is better
        batch_ratio = result.throughput / self.reference["batch_throughput"]
        scores["batch"] = min(batch_ratio * 100, 100)
        
        # Calculate total score (0-1000)
        total_score = sum(scores.values())
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            "inferences_per_watt": (
                result.throughput / result.power_usage if result.power_usage else None
            ),
            "inferences_per_mb": result.throughput / result.memory_usage,
            "ms_per_inference": result.inference_time
        }
        
        return {
            "total_score": round(total_score, 1),
            "subscores": {k: round(v, 1) for k, v in scores.items()},
            "efficiency_metrics": {k: round(v, 2) if v else None 
                                 for k, v in efficiency_metrics.items()}
        }

    def get_performance_tier(self, score: float) -> str:
        """Returns a performance tier based on the score"""
        if score >= 900:
            return "Exceptional (Data Center Grade)"
        elif score >= 800:
            return "Excellent (Server Grade)"
        elif score >= 700:
            return "Very Good (High-End Edge)"
        elif score >= 600:
            return "Good (Edge Device)"
        elif score >= 500:
            return "Fair (Mobile Grade)"
        elif score >= 400:
            return "Basic (IoT Grade)"
        elif score >= 300:
            return "Limited (MCU Grade)"
        else:
            return "Entry Level (CPU Only)"

    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generates a formatted report comparing multiple devices"""
        report = "AI Acceleration Performance Report\n"
        report += "=" * 80 + "\n\n"
        
        # Sort results by total score
        scored_results = []
        for result in results:
            score_data = self.calculate_score(result)
            scored_results.append((result, score_data))
        
        scored_results.sort(key=lambda x: x[1]["total_score"], reverse=True)
        
        # Generate comparison table
        headers = ["Device", "Score", "Tier", "ms/inf", "FPS", "Power(W)"]
        rows = []
        
        for result, score_data in scored_results:
            rows.append([
                result.soc_name,
                f"{score_data['total_score']:.1f}",
                self.get_performance_tier(score_data['total_score']),
                f"{result.inference_time:.1f}",
                f"{result.throughput:.1f}",
                f"{result.power_usage:.1f}" if result.power_usage else "N/A"
            ])
        
        # Format as table
        col_widths = [max(len(str(row[i])) for row in rows + [headers]) 
                     for i in range(len(headers))]
        
        # Add headers
        for i, header in enumerate(headers):
            report += f"{header:<{col_widths[i]}} "
        report += "\n" + "-" * sum(col_widths) + "\n"
        
        # Add rows
        for row in rows:
            for i, item in enumerate(row):
                report += f"{str(item):<{col_widths[i]}} "
            report += "\n"
        
        # Add detailed subscores
        report += "\nDetailed Scores:\n" + "-" * 40 + "\n"
        for result, score_data in scored_results:
            report += f"\n{result.soc_name}:\n"
            for name, subscore in score_data["subscores"].items():
                report += f"  {name:<12}: {subscore:>6.1f}\n"
            
            # Add efficiency metrics
            report += "  Efficiency Metrics:\n"
            for name, value in score_data["efficiency_metrics"].items():
                if value is not None:
                    report += f"    {name:<20}: {value:>6.2f}\n"
        
        return report

def add_scoring_to_benchmark(benchmark_results_path: str, output_path: str):
    """Add scoring to existing benchmark results"""
    with open(benchmark_results_path, 'r') as f:
        data = json.load(f)
    
    scorer = PerformanceScorer()
    results = [BenchmarkResult(**r) for r in data['results']]
    
    # Generate report
    report = scorer.generate_report(results)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    # Add scores to original results
    for result in data['results']:
        result['scores'] = scorer.calculate_score(BenchmarkResult(**result))
    
    # Save updated results
    with open(benchmark_results_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Add performance scoring to benchmark results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to benchmark results JSON file')
    parser.add_argument('--output', type=str, default='performance_report.txt',
                       help='Output path for performance report')
    args = parser.parse_args()

    report = add_scoring_to_benchmark(args.results, args.output)
    print("\nPerformance Report:")
    print(report)

if __name__ == '__main__':
    main()
