"""
Universal CircuitPython ML Benchmark

A standardized benchmark for testing ML matrix operation performance
on microcontrollers and SBCs.

License: See GitHub repository (link below)
"""

# Benchmark metadata
__version__ = "0.2"
__repo__ = "https://github.com/platima/ml-accelerator-benchmark"
__source__ = "https://github.com/platima/ml-accelerator-benchmark/blob/<TODO_commit_hash>/circuitpython-benchmark.py"
__author__ = "Platima"
__firmware__ = "TBC"
__notes__ = ""

import time
import json
import microcontroller
import gc
import board
import analogio
try:
    from ulab import numpy as np
except ImportError:
    # For newer CircuitPython versions
    import ulab.numpy as np

class UniversalBenchmark:
    def __init__(self, channels=3, warmup_runs=2, num_runs=10):
        self.channels = channels
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
        # Setup hardware info
        self.freq = microcontroller.cpu.frequency  # Hz
        self.num_cores = self._detect_core_count()
        
        self._setup_hardware_monitoring()
        
        # Memory tracking
        self.initial_mem = self._get_mem_free()
        
        # Find maximum supported dimensions
        self.max_size = self._find_max_dimensions()
        
        # Store actual benchmark matrices
        self.input_arrays = None
        self.output_array = None
    
    def _detect_core_count(self):
        """Detect number of CPU cores"""
        try:
            return len(microcontroller.cpus)
        except (AttributeError, TypeError):
            return 1  # Default to single core if detection fails
    
    def _setup_hardware_monitoring(self):
        """Setup hardware-specific monitoring"""
        self.has_temp_sensor = True  # CircuitPython always has temperature monitoring
        self.has_power_sensor = False
        
        try:
            # Try to set up voltage monitoring if available
            self.voltage_pin = analogio.AnalogIn(board.VOLTAGE_MONITOR)
            self.has_power_sensor = True
        except (AttributeError, ValueError):
            pass
    
    def _get_mem_free(self):
        """Get available memory in bytes"""
        gc.collect()
        return gc.mem_free()
    
    def _get_temperature(self):
        """Get CPU temperature"""
        return microcontroller.cpu.temperature
    
    def _get_power_usage(self):
        """Get power usage if available"""
        if not self.has_power_sensor:
            return None
            
        try:
            voltage = self.voltage_pin.value * 3.3 / 65536
            return voltage
        except:
            return None
    
    def _calculate_ops(self, size):
        """Calculate the approximate number of floating point operations.
        For matrix multiplication of size NxN:
        - Basic ops per multiplication: N³ (N multiplications and N-1 additions for N² elements)
        - We do this for each channel
        - We also do additional element-wise operations
        """
        # Matrix multiplication ops for each channel
        matmul_ops = size * size * size * self.channels
        
        # Element-wise operations (scaling and addition)
        elementwise_ops = size * size * self.channels * 2
        
        return matmul_ops + elementwise_ops
    
    def _try_create_array(self, size):
        """Try to create test arrays of given size and verify operational memory"""
        try:
            # Get initial memory
            gc.collect()
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
            
            # Clean up
            del arrays
            del out_arr
            gc.collect()
                
            return True
        except:
            return False
    
    def _find_max_dimensions(self):
        """Binary search for maximum supported dimensions"""
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
                    print("Stopping at 256, although we could go higher!")
                    final_size = size
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
        return final_size - 15 # TODO this is to avoid a memory error that I'm yet to deal with
    
    def _create_benchmark_arrays(self):
        """Create actual arrays for benchmarking"""
        self.input_arrays = []
        gc.collect()
        for _ in range(self.channels):
            arr = np.zeros((self.max_size, self.max_size))
            arr += np.linspace(0, 1, arr.size).reshape(arr.shape)  # Create gradient
            self.input_arrays.append(arr)
        
        self.output_array = np.zeros((self.max_size, self.max_size))
    
    def _run_math_ops(self):
        """Run actual mathematical operations with careful memory management"""
        ops_time = time.monotonic_ns()  # Use nanosecond precision
        
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
        
        elapsed_ns = time.monotonic_ns() - ops_time
        return elapsed_ns / 1_000_000  # Convert to milliseconds
    
    def run(self):
        """Run complete benchmark with multicore awareness"""
        print("\nStarting benchmark...")
        print("CPU Frequency: {:.1f} MHz".format(self.freq / 1_000_000))
        print("Number of CPU cores: {}".format(self.num_cores))
        print("Initial free memory: {} bytes".format(self.initial_mem))
        print("Array size per channel: {}x{}".format(self.max_size, self.max_size))
        
        # Warn about potential multicore impacts
        if self.num_cores > 1:
            print("Note: Multicore system detected. Benchmark results may be affected by:")
            print("- Background tasks on other cores")
            print("- Shared memory bandwidth")
            print("- Cache coherency overhead")
        
        try:
            self._create_benchmark_arrays()
        except Exception as e:
            print("Error: Failed to create benchmark arrays: {}".format(str(e)))
            return None
        
        mem_after_data = self._get_mem_free()
        print("Free memory after data creation: {} bytes".format(mem_after_data))
        print("Data size: {} bytes".format(self.initial_mem - mem_after_data))
        
        # Warmup runs
        print("Performing warmup runs...")
        for _ in range(self.warmup_runs):
            self._run_math_ops()
        
        # Benchmark runs
        print("Running {} iterations...".format(self.num_runs))
        times = []
        temps = []
        mems = []
        power = []
        
        for i in range(self.num_runs):
            print("Progress: {}/{}".format(i+1, self.num_runs))
            
            # Run benchmark and collect metrics
            runtime = self._run_math_ops()
            times.append(runtime)
            
            temp = self._get_temperature()
            if temp is not None:
                temps.append(temp)
                
            power_usage = self._get_power_usage()
            if power_usage is not None:
                power.append(power_usage)
                
            mems.append(self._get_mem_free())
        
        # Calculate results
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate total operations and normalized performance metrics
        total_ops = self._calculate_ops(self.max_size)
        ops_per_second = total_ops / (avg_time / 1000)  # Convert ms to seconds
        ops_per_second_per_mhz = ops_per_second / (self.freq / 1_000_000)
        # Calculate theoretical total power (efficiency × MHz × cores)
        theoretical_power = ops_per_second_per_mhz * (self.freq / 1_000_000) * self.num_cores
        
        # Get current date in ISO format
        current_date = time.localtime()
        date_str = "{:04d}-{:02d}-{:02d}".format(
            current_date[0], current_date[1], current_date[2]
        )
        
        results = {
            "_meta": {
                "Source version": __version__,
                "Source code": __source__,
                "Source repo": __repo__,
                "Test date": date_str,
                "Tester": __author__,
                "Firmware": __firmware__,
                "Notes": __notes__
            },
            "device": {
                "cpu_freq_mhz": self.freq / 1_000_000,
                "board_type": self._detect_board_type(),
                "temp_sensor": self.has_temp_sensor,
                "power_sensor": self.has_power_sensor,
                "num_cores": self.num_cores
            },
            "performance": {
                "channels": self.channels,
                "array_size": self.max_size,
                "memory_total": self.initial_mem,
                "memory_used": self.initial_mem - min(mems),
                "min_inference_ms": min_time,
                "max_inference_ms": max_time,
                "avg_inference_ms": avg_time,
                "throughput_fps": 1000 / avg_time
            },
            "benchmark": {
                "total_ops": total_ops,
                "ops_per_second": ops_per_second,
                "normalized_score": ops_per_second_per_mhz,
                "theoretical_power": theoretical_power
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
        try:
            import esp32
            board_type = "esp32"
            del esp32
            gc.collect()
            return board_type
        except:
            try:
                import rp2 # TODO this does not exist in CircuitPython
                # Both have rp2 module, differentiate by frequency
                board_type = "rp2350" if self.freq >= 133_000_000 else "rp2040"
                del rp2
                gc.collect()
                return board_type
            except:
                # Fall back to other detection methods
                if self.freq >= 133_000_000:
                    return "unknown_highfreq"
                return "unknown_lowfreq"
        
    def format_number(self, n):
        """Format a number for JSON output without scientific notation"""
        if isinstance(n, bool):
            return "true" if n else "false"
        elif isinstance(n, int):
            return str(n)
        elif isinstance(n, float):
            if abs(n) >= 1e5:
                # For large numbers, use fixed notation with underscore separators
                return "{:.3f}".format(n).replace(",", "")
            else:
                # For smaller numbers, use regular float notation
                return "{:.3f}".format(n)
        else:
            return '"{}"'.format(n)  # String values get quoted

    def print_results(self, results):
        """Print results in nicely formatted JSON to stdout"""
        try:
            meta_order = [
                "Source version",
                "Source code",
                "Source repo",
                "Test date",
                "Tester",
                "Firmware",
                "Notes"
            ]
            
            device_order = [
                "board_type",
                "cpu_freq_mhz",
                "num_cores",
                "temp_sensor",
                "power_sensor"
            ]
            
            performance_order = [
                "channels",
                "array_size",
                "memory_total",
                "memory_used",
                "min_inference_ms",
                "max_inference_ms",
                "avg_inference_ms",
                "throughput_fps"
            ]
            
            if "avg_temperature" in results["performance"]:
                performance_order.extend(["avg_temperature", "max_temperature"])
            
            benchmark_order = [
                "total_ops",
                "ops_per_second",
                "normalized_score",
                "theoretical_power"
            ]
            
            # Build JSON string with proper formatting
            output = []
            output.append("{")
            
            # Metadata section
            output.append('    "_meta": {')
            for i, key in enumerate(meta_order):
                comma = "," if i < len(meta_order) - 1 else ""
                output.append('        "{}": {}{}'.format(key, self.format_number(results["_meta"][key]), comma))
            output.append("    },")
            
            # Device section
            output.append('    "device": {')
            for i, key in enumerate(device_order):
                comma = "," if i < len(device_order) - 1 else ""
                output.append('        "{}": {}{}'.format(key, self.format_number(results["device"][key]), comma))
            output.append("    },")
            
            # Performance section
            output.append('    "performance": {')
            for i, key in enumerate(performance_order):
                if key in results["performance"]:
                    comma = "," if i < len([k for k in performance_order if k in results["performance"]]) - 1 else ""
                    output.append('        "{}": {}{}'.format(key, self.format_number(results["performance"][key]), comma))
            output.append("    },")
            
            # Benchmark section
            output.append('    "benchmark": {')
            for i, key in enumerate(benchmark_order):
                comma = "," if i < len(benchmark_order) - 1 else ""
                output.append('        "{}": {}{}'.format(key, self.format_number(results["benchmark"][key]), comma))
            output.append("    }")
            
            output.append("}")
            
            # Print with newlines
            print("\nResults:")
            print("\n".join(output))
            
        except Exception as e:
            print("Error formatting results:", str(e))
            print("Raw results:", str(results))


# Example usage
if __name__ == "__main__":
    try:
        # Create and run benchmark
        benchmark = UniversalBenchmark(
            channels=3,
            warmup_runs=2,
            num_runs=10
        )
        
        results = benchmark.run()
        if results:
            benchmark.print_results(results)
            
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
