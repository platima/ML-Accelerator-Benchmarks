"""
Results Handler Module - MicroPython Compatible
Handles saving, loading, and validating benchmark results
"""
import os
import json
import gc
from time import localtime

try:
    from datetime import datetime
    MICROPYTHON = False
except ImportError:
    MICROPYTHON = True

class SimpleJSONEncoder:
    """Minimal JSON encoder for MicroPython"""
    @staticmethod
    def encode(obj, indent=None):
        if isinstance(obj, dict):
            items = [f'"{k}": {SimpleJSONEncoder.encode(v)}' 
                    for k, v in obj.items()]
            return "{" + ", ".join(items) + "}"
        elif isinstance(obj, (list, tuple)):
            items = [SimpleJSONEncoder.encode(x) for x in obj]
            return "[" + ", ".join(items) + "]"
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, bool):
            return str(obj).lower()
        elif obj is None:
            return "null"
        else:
            return f'"{str(obj)}"'

class ResultsHandler:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self._ensure_dir_exists()
        
    def _ensure_dir_exists(self):
        """Create results directory if it doesn't exist"""
        try:
            os.mkdir(self.results_dir)
        except:
            pass
            
    def _get_timestamp(self):
        """Get formatted timestamp string"""
        if MICROPYTHON:
            t = localtime()
            return f"{t[0]}-{t[1]:02d}-{t[2]:02d}"
        else:
            return datetime.now().strftime("%Y-%m-%d")
            
    def _sanitize_filename(self, name):
        """Create safe filename from device name"""
        # Remove or replace unsafe characters
        unsafe = '<>:"/\\|?*'
        for char in unsafe:
            name = name.replace(char, '_')
        return name
            
    def save_result(self, results, device_name, description=None):
        """Save benchmark results to a file"""
        # Add metadata
        results["_metadata"] = {
            "timestamp": self._get_timestamp(),
            "device": device_name
        }
        
        # Create filename
        timestamp = self._get_timestamp()
        safe_name = self._sanitize_filename(device_name)
        filename = f"{timestamp}_{safe_name}"
        if description:
            filename += f"_{self._sanitize_filename(description)}"
        filename += ".json"
        
        # Save results
        filepath = os.path.join(self.results_dir, filename)
        try:
            with open(filepath, 'w') as f:
                if MICROPYTHON:
                    f.write(SimpleJSONEncoder.encode(results))
                else:
                    json.dump(results, f, indent=2)
                    
            return filepath
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return None
        finally:
            gc.collect()  # Clean up
            
    def load_result(self, filename):
        """Load benchmark results from a file"""
        filepath = os.path.join(self.results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                if MICROPYTHON:
                    # Simple JSON parser for MicroPython
                    text = f.read()
                    # Use eval() with strict constraints for simple JSON parsing
                    # Note: This is safe for our known result format
                    # but should be replaced with proper JSON parsing
                    # in production environments
                    results = eval(text, {"__builtins__": {}}, {})
                else:
                    results = json.load(f)
                return results
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return None
        finally:
            gc.collect()
            
    def list_results(self, filter_str=None):
        """List available result files"""
        results = []
        try:
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    if filter_str is None or filter_str in filename:
                        results.append(filename)
        except Exception as e:
            print(f"Error listing results: {str(e)}")
        return sorted(results)
        
    def get_latest_result(self, device_name=None):
        """Get most recent benchmark result"""
        results = self.list_results(device_name)
        if not results:
            return None
        return self.load_result(results[-1])
        
    def compare_results(self, filenames):
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
                if result:
                    comparison["devices"].append(
                        result.get("_metadata", {}).get("device", "Unknown")
                    )
                    
                    perf = result.get("performance", {})
                    for metric in comparison["metrics"]:
                        comparison["metrics"][metric].append(
                            perf.get(metric)
                        )
                        
            except Exception as e:
                print(f"Error comparing {filename}: {str(e)}")
                
        return comparison
        
    def clear_old_results(self, keep_days=30):
        """Remove old result files to free up space"""
        if MICROPYTHON:
            print("clear_old_results not supported in MicroPython")
            return
            
        current = datetime.now()
        for filename in self.list_results():
            try:
                # Extract date from filename (YYYY-MM-DD format)
                date_str = filename.split('_')[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                # Remove if older than keep_days
                if (current - file_date).days > keep_days:
                    os.remove(os.path.join(self.results_dir, filename))
            except:
                continue

def main():
    """Example usage"""
    handler = ResultsHandler()
    
    # Example results
    example_result = {
        "performance": {
            "inference_time_ms": 15.5,
            "throughput_fps": 64.5,
            "memory_usage_kb": 1024
        }
    }
    
    # Save example
    handler.save_result(example_result, "Test Device")
    
    # List results
    print("\nAvailable Results:")
    for result in handler.list_results():
        print(f"- {result}")
        
    # Load latest
    latest = handler.get_latest_result()
    if latest:
        print("\nLatest Result:")
        if MICROPYTHON:
            print(SimpleJSONEncoder.encode(latest))
        else:
            print(json.dumps(latest, indent=2))

if __name__ == "__main__":
    main()