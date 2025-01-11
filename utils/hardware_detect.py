"""
Universal ML Hardware Detection - MicroPython Compatible
Automatically detects available ML acceleration capabilities
"""
import os
import sys
import gc

try:
    import platform
    MICROPYTHON = False
except ImportError:
    MICROPYTHON = True
    try:
        import machine
    except ImportError:
        machine = None

class HardwareDetector:
    def __init__(self):
        self.platform = self._detect_platform()
        self.capabilities = self._detect_capabilities()
        
    def _detect_platform(self):
        """Detect basic platform information"""
        info = {
            "os": "unknown",
            "architecture": "unknown",
            "python_impl": "MicroPython" if MICROPYTHON else "CPython",
            "is_micropython": MICROPYTHON
        }
        
        if MICROPYTHON:
            # MicroPython-specific detection
            try:
                if machine:
                    info["architecture"] = machine.unique_id()
                    # Try to detect board type
                    board_name = None
                    try:
                        # Different methods to detect board type
                        if hasattr(machine, 'board'):
                            board_name = machine.board
                        elif os.uname().machine:
                            board_name = os.uname().machine
                    except:
                        pass
                    info["board"] = board_name or "unknown"
                    
                # Try to get OS information
                try:
                    info["os"] = os.uname().sysname
                except:
                    pass
            except:
                pass
        else:
            # CPython platform detection
            info["os"] = platform.system()
            info["architecture"] = platform.machine()
            
        return info
        
    def _detect_capabilities(self):
        """Detect available ML acceleration capabilities"""
        caps = {
            "accelerators": [],
            "memory_mb": self._get_total_memory(),
            "supported_ops": [],
            "quantization": []
        }
        
        if MICROPYTHON:
            # MicroPython-specific hardware detection
            self._detect_micropython_capabilities(caps)
        else:
            # Standard hardware detection
            self._detect_standard_capabilities(caps)
            
        return caps
        
    def _detect_micropython_capabilities(self, caps):
        """Detect MicroPython-specific capabilities"""
        if not machine:
            return
            
        try:
            # Check for hardware features
            if hasattr(machine, 'freq'):
                freq = machine.freq()
                caps["cpu_freq"] = freq
                
            # Add CPU info
            caps["accelerators"].append({
                "type": "CPU",
                "model": self._detect_micropython_cpu(),
                "features": self._detect_micropython_features()
            })
            
            # Check for specialized hardware
            if self._check_esp32():
                caps["accelerators"].append({
                    "type": "ESP32",
                    "features": ["ULP", "WiFi", "BLE"]
                })
                caps["quantization"].extend(["int8"])
                
            elif self._check_rp2040():
                caps["accelerators"].append({
                    "type": "RP2040",
                    "features": ["PIO", "DMA"]
                })
                
            # Add basic op support
            caps["supported_ops"].extend([
                "basic_arithmetic",
                "digital_io",
                "analog_io",
                "pwm"
            ])
            
        except Exception as e:
            print(f"Error detecting capabilities: {str(e)}")
            
    def _detect_micropython_cpu(self):
        """Detect MicroPython CPU details"""
        try:
            if machine:
                # Try different methods to get CPU info
                if hasattr(os, 'uname'):
                    return os.uname().machine
                elif hasattr(machine, 'unique_id'):
                    # Convert unique_id to readable format
                    uid = machine.unique_id()
                    if isinstance(uid, bytes):
                        return 'CPU-' + ''.join(f'{x:02x}' for x in uid[-4:])
        except:
            pass
        return "Unknown MCU"
        
    def _detect_micropython_features(self):
        """Detect MicroPython-specific features"""
        features = []
        if not machine:
            return features
            
        try:
            # Check for common hardware features
            if hasattr(machine, 'ADC'):
                features.append('ADC')
            if hasattr(machine, 'PWM'):
                features.append('PWM')
            if hasattr(machine, 'I2C'):
                features.append('I2C')
            if hasattr(machine, 'SPI'):
                features.append('SPI')
                
            # Check for specialized features
            if hasattr(machine, 'RTC'):
                features.append('RTC')
            if hasattr(machine, 'WDT'):
                features.append('WDT')
                
            # Check for networking
            if 'network' in sys.modules:
                features.append('NETWORK')
                
        except:
            pass
            
        return features
        
    def _check_esp32(self):
        """Check if running on ESP32"""
        try:
            return "ESP32" in os.uname().machine
        except:
            return False
            
    def _check_rp2040(self):
        """Check if running on RP2040"""
        try:
            return "RP2" in os.uname().machine
        except:
            return False
        
    def _detect_standard_capabilities(self, caps):
        """Detect standard Python hardware capabilities"""
        # CPU Detection
        caps["accelerators"].append({
            "type": "CPU",
            "model": self._detect_cpu_model(),
            "features": self._detect_cpu_features()
        })
        
        # Check for specialized hardware
        self._check_npu(caps)
        self._check_tpu(caps)
        self._check_edge_tpu(caps)
        
    def _detect_cpu_model(self):
        """Detect CPU model"""
        if MICROPYTHON:
            return "MicroController"
            
        try:
            if sys.platform == "linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            return platform.processor() or "Unknown"
        except:
            return "Unknown"
            
    def _detect_cpu_features(self):
        """Detect CPU features"""
        features = []
        if MICROPYTHON:
            return self._detect_micropython_features()
            
        try:
            if sys.platform == "linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "flags" in line:
                            flags = line.split(":")[1].strip().split()
                            if "neon" in flags:
                                features.append("NEON")
                            if "sse" in flags:
                                features.append("SSE")
                            if "avx" in flags:
                                features.append("AVX")
        except:
            pass
        return features
        
    def _get_total_memory(self):
        """Get total memory in MB"""
        if MICROPYTHON:
            try:
                gc.collect()
                free = gc.mem_free()
                alloc = gc.mem_alloc()
                return (free + alloc) // (1024 * 1024)  # Convert to MB
        except:
            return 0
            
        else:
            try:
                import psutil
                return psutil.virtual_memory().total // (1024 * 1024)
            except:
                return 0
                
    def _check_npu(self, caps):
        """Check for Neural Processing Unit"""
        try:
            # Check for common NPU drivers/libraries
            if os.path.exists("/dev/mali0"):
                caps["accelerators"].append({
                    "type": "NPU",
                    "model": "Mali NPU",
                    "ops": ["conv2d", "matmul"]
                })
                caps["quantization"].extend(["int8", "int16"])
            elif os.path.exists("/dev/mtk_mdla"):
                caps["accelerators"].append({
                    "type": "NPU",
                    "model": "MediaTek APU",
                    "ops": ["conv2d", "matmul"]
                })
                caps["quantization"].extend(["int8"])
        except:
            pass
            
    def _check_tpu(self, caps):
        """Check for Tensor Processing Unit"""
        try:
            import importlib
            if importlib.util.find_spec("tflite_runtime"):
                # Check for Edge TPU
                if os.path.exists("/dev/apex_0"):
                    caps["accelerators"].append({
                        "type": "TPU",
                        "model": "Edge TPU",
                        "ops": ["conv2d", "matmul"]
                    })
                    caps["quantization"].extend(["int8"])
        except:
            pass
            
    def _check_edge_tpu(self, caps):
        """Check for Google Edge TPU"""
        try:
            import importlib
            if importlib.util.find_spec("pycoral"):
                caps["accelerators"].append({
                    "type": "TPU",
                    "model": "Google Edge TPU",
                    "ops": ["conv2d", "matmul", "pooling"],
                    "quantization": ["int8"]
                })
                caps["quantization"].extend(["int8"])
        except:
            pass
            
    def get_recommended_config(self):
        """Get recommended benchmark configuration"""
        config = {
            "framework": "tflite-micro" if MICROPYTHON else "tflite",
            "quantization": [],
            "batch_size": 1,
            "threads": 1
        }
        
        # Adjust based on capabilities
        if MICROPYTHON:
            if self._check_esp32():
                config["framework"] = "tflite-micro"
                config["quantization"] = ["int8"]
            elif self._check_rp2040():
                config["framework"] = "cmsis-nn"
                config["quantization"] = ["int8"]
        else:
            for acc in self.capabilities["accelerators"]:
                if acc["type"] == "NPU":
                    config["framework"] = "tflite"
                    config["quantization"] = ["int8"]
                    break
                elif acc["type"] == "TPU":
                    config["framework"] = "tflite-edge-tpu"
                    config["quantization"] = ["int8"]
                    break
                    
        return config
        
    def print_capabilities(self):
        """Print detected capabilities in a readable format"""
        print("\nHardware Detection Results")
        print("=" * 20)
        
        # Platform info
        print(f"Platform: {self.platform['os']}")
        print(f"Python: {self.platform['python_impl']}")
        if MICROPYTHON:
            print(f"Board: {self.platform.get('board', 'Unknown')}")
        print(f"Memory: {self.capabilities['memory_mb']} MB")
        
        # Accelerators
        print("\nAccelerators:")
        for acc in self.capabilities["accelerators"]:
            print(f"- Type: {acc['type']}")
            print(f"  Model: {acc.get('model', 'Unknown')}")
            if "features" in acc:
                print(f"  Features: {', '.join(acc['features'])}")
            if "ops" in acc:
                print(f"  Ops: {', '.join(acc['ops'])}")
                
        # Quantization support
        if self.capabilities["quantization"]:
            print("\nQuantization Support:")
            print(", ".join(self.capabilities["quantization"]))
            
        # Memory management
        gc.collect()

def main():
    """Example usage"""
    detector = HardwareDetector()
    detector.print_capabilities()
    
    config = detector.get_recommended_config()
    print("\nRecommended Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()