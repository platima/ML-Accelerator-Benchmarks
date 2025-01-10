"""
Universal MicroPython ML Benchmark
Compatible with various SBCs and microcontrollers
Uses minimal dependencies for wide compatibility
"""
import time
import json
import machine
import gc
from ulab import numpy as np

class UniversalBenchmark:
    def __init__(self, channels=3, warmup_runs=2, num_runs=10):
        self.channels = channels
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
        # Setup hardware info
        self.freq = machine.freq() // 1_000_000  # MHz
        self._setup_hardware_monitoring()
        
        # Memory tracking
        self.initial_mem = self._get_mem_free()
        
        # Find maximum supported dimensions
        self.max_size = self._find_max_dimensions()
        
        # Store actual benchmark matrices
        self.input_arrays = None
        self.output_array = None
        
    def _setup_hardware_monitoring(self):
        """Setup hardware-specific monitoring"""
        self.has_temp_sensor = False
        self.has_power_sensor = False
        
        try:
            # Try RP2040/RP2350 temperature sensor
            self.temp_sensor = machine.ADC(4)
            self.has_temp_sensor = True
        except:
            pass
            
        try:
            # Try voltage monitoring if available
            machine.ADC(29)  # Voltage monitoring pin varies by board
            self.has_power_sensor = True
        except:
            pass
    
    def _get_mem_free(self):
        """Get available memory in bytes"""
        gc.collect()
        return gc.mem_free()
    
    def _get_temperature(self):
        """Get CPU temperature if available"""
        if not self.has_temp_sensor:
            return None
            
        try:
            # RP2040/RP2350 method
            adc = self.temp_sensor.read_u16() * 3.3 / 65535
            return 27 - (adc - 0.706) / 0.001721
        except:
            return None
    
    def _get_power_usage(self):
        """Get power usage if available"""
        if not self.has_power_sensor:
            return None
            
        try:
            # Implementation varies by board
            return None  # TODO: Implement for specific boards
        except:
            return None
    
    def _try_create_array(self, size):
        """Try to create test arrays of given size and verify operational memory"""
        try:
            # Get initial memory
            initial_mem = self._get_mem_free()
            arrays = []
            
            # Test input arrays
            for _ in range(self.channels):
                arr = np.zeros((size, size))
                arr += 0.1  # Add some non-zero data
                arrays.append(arr)
            
            # Test output array
            out_arr = np.zeros((size, size))
            
            # Test operations memory
            try:
                # Test matrix multiplication (needs temporary memory)
                test_result = np.dot(arrays[0], arrays[0])
                # Test element-wise ops
                test_result = np.sin(test_result)
                # Force cleanup
                del test_result
                gc.collect()
            except:
                return False
            
            # Check memory usage isn't too high (keep 40% free for operations)
            mem_used = initial_mem - self._get_mem_free()
            if mem_used > initial_mem * 0.6:
                return False
            
            # Clean up
            del arrays
            del out_arr
            gc.collect()
                
            return True
        except:
            return False
    
    def _find_max_dimensions(self):
        """Binary search for maximum supported dimensions with clear feedback"""
        print("\nFinding maximum supported dimensions...")
        print("--------------------------------------")
        
        # Initial phase: double until failure
        size = 16
        last_success = None
        
        print("Phase 1: Finding upper bound")
        while True:
            print(f"Testing {size}x{size}...", end=" ")
            if self._try_create_array(size):
                print("OK")
                last_success = size
                if size >= 256:  # Upper limit for most microcontrollers
                    break
                size *= 2
            else:
                print("FAILED")
                break
        
        if last_success is None:
            print("Could not find valid starting size!")
            return 8  # Return minimum practical size
            
        print(f"\nPhase 2: Binary search between {last_success} and {size}")
        
        # Binary search between last success and failure
        low = last_success
        high = size
        
        while low < high - 1:
            mid = (low + high) // 2
            print(f"Testing {mid}x{mid}...", end=" ")
            
            if self._try_create_array(mid):
                print("OK")
                low = mid
            else:
                print("FAILED")
                high = mid
        
        final_size = low
        print("\nResults:")
        print(f"Maximum stable size: {final_size}x{final_size}")
        print("--------------------------------------")
        return final_size
    
    def _create_benchmark_arrays(self):
        """Create actual arrays for benchmarking"""
        self.input_arrays = []
        for _ in range(self.channels):
            arr = np.zeros((self.max_size, self.max_size))
            arr += np.linspace(0, 1, arr.size).reshape(arr.shape)  # Create gradient
            self.input_arrays.append(arr)
        
        self.output_array = np.zeros((self.max_size, self.max_size))
    
    def _run_math_ops(self):
        """Run actual mathematical operations with careful memory management"""
        ops_time = time.ticks_ms()
        
        # Reset output array manually
        self.output_array = np.zeros(self.output_array.shape)
        
        # Process one channel at a time
        for i in range(self.channels):
            # Matrix multiplication with temp array
            result = np.dot(self.input_arrays[i], self.input_arrays[i])
            
            # Simple scaling (memory efficient)
            result *= 0.5
            
            # Add to output
            self.output_array += result
            
            # Force cleanup
            del result
            gc.collect()
        
        # Final normalization (simple scaling)
        self.output_array *= (1.0 / self.channels)
        
        return time.ticks_diff(time.ticks_ms(), ops_time)
        
        return time.ticks_diff(time.ticks_ms(), ops_time)
    
    def run(self):
        """Run complete benchmark"""
        print("\nStarting benchmark...")
        print(f"CPU Frequency: {self.freq} MHz")
        print(f"Initial free memory: {self.initial_mem} bytes")
        print(f"Array size per channel: {self.max_size}x{self.max_size}")
        
        try:
            self._create_benchmark_arrays()
        except:
            print("Error: Failed to create benchmark arrays")
            return None
        
        mem_after_data = self._get_mem_free()
        print(f"Free memory after data creation: {mem_after_data} bytes")
        print(f"Data size: {self.initial_mem - mem_after_data} bytes")
        
        # Warmup runs
        print("Performing warmup runs...")
        for _ in range(self.warmup_runs):
            self._run_math_ops()
        
        # Benchmark runs
        print(f"Running {self.num_runs} iterations...")
        times = []
        temps = []
        mems = []
        power = []
        
        for i in range(self.num_runs):
            print(f"Progress: {i+1}/{self.num_runs}")
            
            # Run benchmark and collect metrics
            runtime = self._run_math_ops()
            times.append(runtime)
            
            temp = self._get_temperature()
            if temp:
                temps.append(temp)
                
            power_usage = self._get_power_usage()
            if power_usage:
                power.append(power_usage)
                
            mems.append(self._get_mem_free())
        
        # Calculate results
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results = {
            "device": {
                "cpu_freq_mhz": self.freq,
                "board_type": self._detect_board_type(),
                "temp_sensor": self.has_temp_sensor,
                "power_sensor": self.has_power_sensor
            },
            "performance": {
                "avg_inference_ms": avg_time,
                "min_inference_ms": min_time,
                "max_inference_ms": max_time,
                "throughput_fps": 1000 / avg_time,
                "memory_total": self.initial_mem,
                "memory_used": self.initial_mem - min(mems),
                "array_size": self.max_size,
                "channels": self.channels
            }
        }
        
        # Add temperature if available
        if temps:
            results["performance"].update({
                "avg_temperature": sum(temps) / len(temps),
                "max_temperature": max(temps)
            })
            
        # Add power if available
        if power:
            results["performance"].update({
                "avg_power_usage": sum(power) / len(power),
                "max_power_usage": max(power)
            })
        
        return results
    
    def _detect_board_type(self):
        """Try to detect the board type"""
        # Start with frequency-based detection
        if self.freq >= 133:
            if self.has_temp_sensor:
                return "rp2350"  # Newer Pico
            return "unknown_highfreq"
        else:
            if self.has_temp_sensor:
                return "rp2040"  # Original Pico
            return "unknown_lowfreq"
    
    def save_results(self, results, filename='benchmark_results.json'):
        """Save results to JSON file with nice formatting"""
        try:
            def format_value(v):
                if isinstance(v, bool):
                    return str(v).lower()
                if isinstance(v, (int, float)):
                    if isinstance(v, float):
                        return f"{v:.3f}"  # Format floats to 3 decimal places
                    return str(v)
                return f'"{v}"'

            # Manual pretty printing
            json_str = "{\n"
            
            # Device section
            json_str += '    "device": {\n'
            dev = results["device"]
            dev_items = list(dev.items())
            for i, (k, v) in enumerate(dev_items):
                json_str += f'        "{k}": {format_value(v)}'
                if i < len(dev_items) - 1:
                    json_str += ','
                json_str += '\n'
            json_str += '    },\n'
            
            # Performance section
            json_str += '    "performance": {\n'
            perf = results["performance"]
            perf_items = list(perf.items())
            for i, (k, v) in enumerate(perf_items):
                json_str += f'        "{k}": {format_value(v)}'
                if i < len(perf_items) - 1:
                    json_str += ','
                json_str += '\n'
            json_str += '    }\n'
            
            json_str += '}'

            # Save to file
            with open(filename, 'w') as f:
                f.write(json_str)
            
            # Display results
            print(f"\nResults saved to {filename}")
            print("\nResults:")
            print(json_str)
            
        except Exception as e:
            print("Error saving results to file:", str(e))
            print("Raw results:", results)

# Example usage
if __name__ == '__main__':
    try:
        # Create and run benchmark
        benchmark = UniversalBenchmark(
            channels=3,
            warmup_runs=2,
            num_runs=10
        )
        
        results = benchmark.run()
        if results:
            benchmark.save_results(results)
            
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
