"""
Results Handler Module
Handles saving, loading, and validating benchmark results
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import jsonschema

class ResultsHandler:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict:
        """Load results schema from file"""
        schema_path = self.results_dir / "results-schema.json"
        try:
            with open(schema_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load schema: {e}")
            return {}
            
    def save_result(self, 
                   results: Dict,
                   device_name: str,
                   description: Optional[str] = None) -> str:
        """Save benchmark results to a file"""
        # Validate results against schema
        if self.schema:
            try:
                jsonschema.validate(instance=results, schema=self.schema)
            except jsonschema.exceptions.ValidationError as e:
                print(f"Warning: Results do not match schema: {e}")
        
        # Create filename with timestamp and device name
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{timestamp} {device_name}"
        if description:
            filename += f" {description}"
        filename += ".json"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save results
        file_path = self.results_dir / filename
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return str(file_path)
        
    def load_result(self, filename: str) -> Dict:
        """Load benchmark results from a file"""
        file_path = self.results_dir / filename
        try:
            with open(file_path) as f:
                results = json.load(f)
                
            # Validate loaded results
            if self.schema:
                jsonschema.validate(instance=results, schema=self.schema)
                
            return results
        except Exception as e:
            raise RuntimeError(f"Error loading results from {filename}: {e}")
            
    def list_results(self, filter_str: Optional[str] = None) -> List[str]:
        """List available result files"""
        results = []
        for file in self.results_dir.glob("*.json"):
            if file.name == "results-schema.json":
                continue
            if filter_str is None or filter_str in file.name:
                results.append(file.name)
        return sorted(results)
        
    def compare_results(self, filenames: List[str]) -> Dict:
        """Compare multiple benchmark results"""
        comparison = {
            "devices": [],
            "metrics": {
                "inference_time_ms": [],
                "throughput_fps": [],
                "memory_usage_kb": []
            }
        }
        
        for filename in filenames:
            try:
                result = self.load_result(filename)
                comparison["devices"].append(result["device"]["name"])
                
                # Extract common metrics
                for metric in comparison["metrics"].keys():
                    if metric in result["performance"]:
                        comparison["metrics"][metric].append(
                            result["performance"][metric])
                    else:
                        comparison["metrics"][metric].append(None)
                        
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                
        return comparison
        
    def get_latest_result(self, device_name: Optional[str] = None) -> Optional[Dict]:
        """Get most recent benchmark result"""
        results = self.list_results(device_name)
        if not results:
            return None
            
        # Latest result will be last when sorted
        return self.load_result(results[-1])
        
    def aggregate_results(self, device_type: str) -> Dict:
        """Aggregate results for a specific device type"""
        aggregated = {
            "device_type": device_type,
            "samples": 0,
            "metrics": {}
        }
        
        # Find all results for device type
        results = self.list_results(device_type)
        if not results:
            return aggregated
            
        # Collect metrics
        for filename in results:
            try:
                result = self.load_result(filename)
                aggregated["samples"] += 1
                
                # Update metrics
                for metric, value in result["performance"].items():
                    if metric not in aggregated["metrics"]:
                        aggregated["metrics"][metric] = []
                    aggregated["metrics"][metric].append(value)
                    
            except Exception as e:
                print(f"Warning: Error processing {filename}: {e}")
                
        # Calculate statistics
        for metric, values in aggregated["metrics"].items():
            if values:
                aggregated["metrics"][metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
                
        return aggregated

def main():
    """Example usage"""
    handler = ResultsHandler()
    
    # List available results
    print("\nAvailable Results:")
    for result in handler.list_results():
        print(f"- {result}")
        
    # Compare latest results
    results = handler.list_results()
    if len(results) >= 2:
        print("\nComparison of latest results:")
        comparison = handler.compare_results(results[-2:])
        print(json.dumps(comparison, indent=2))

if __name__ == "__main__":
    main()