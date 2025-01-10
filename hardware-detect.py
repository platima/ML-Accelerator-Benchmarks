"""
Universal ML Hardware Detection
Automatically detects available ML acceleration capabilities
"""
import os
import sys
import subprocess
from typing import Dict, Optional, List
import json
import platform

class HardwareDetector:
    def __init__(self):
        self.platform = self._detect_platform()
        self.capabilities = self._detect_capabilities()
        
    def _detect_platform(self) -> Dict:
        """Detect basic platform information"""
        info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_impl": platform.python_implementation(),
            "is_micropython": platform.python_implementation() == "MicroPython"
        }
        
        # Detect if running on Android
        if os.path.exists("/system/build.prop"):
            info["os"] = "Android"
            # TODO: Add MediaTek APU / Snapdragon NPU detection
            
        return info
        
    def _detect_capabilities(self) -> Dict:
        """Detect available ML acceleration capabilities"""
        caps = {
            "accelerators": [],
            "memory_mb": self._get_total_memory(),
            "supported_ops": [],
            "quantization": []
        }
        
        # Check for RKNN
        if self._check_rknn():
            caps["accelerators"].append({
                "type": "RKNN",
                "model": self._detect_rockchip_model(),
                "ops": ["conv2d", "matmul", "pooling"]  # TODO: Get actual supported ops
            })
            caps["quantization"].extend(["int8", "uint8", "int16"])
            
        # Check for TPU-MLIR
        if self._check_cvitek():
            caps["accelerators"].append({
                "type": "CVITEK",
                "model": self._detect_cvitek_model(),
                "ops": ["conv2d", "matmul"]  # TODO: Get actual supported ops
            })
            caps["quantization"].extend(["int8", "uint8"])
            
        # Check for TinyML platforms
        if self._check_tinyml():
            caps["accelerators"].append({
                "type": "TINYML",
                "model": self._detect_tinyml_model(),
                "ops": ["quantized_conv2d", "quantized_matmul"]
            })
            caps["quantization"].extend(["int8"])
            
        # Check for FPGA
        fpga_info = self._check_fpga()
        if fpga_info:
            caps["accelerators"].append({
                "type": "FPGA",
                "model": fpga_info["model"],
                "ops": fpga_info["ops"]
            })
            
        # Always add CPU capabilities
        caps["accelerators"].append({
            "type": "CPU",
            "model": self._detect_cpu_model(),
            "features": self._detect_cpu_features()
        })
        
        return caps
        
    def _check_rknn(self) -> bool:
        """Check for RKNN capability"""
        try:
            import rknn.api
            return True
        except ImportError:
            return False
            
    def _check_cvitek(self) -> bool:
        """Check for CVITEK TPU-MLIR capability"""
        try:
            import cvi_toolkit
            return True
        except ImportError:
            return False
            
    def _check_tinyml(self) -> Optional[Dict]:
        """Check for TinyML platforms"""
        # TODO: Implement TinyML platform detection
        # Arduino Nano 33 BLE, etc
        return None
        
    def _check_fpga(self) -> Optional[Dict]:
        """Check for FPGA accelerators"""
        # TODO: Implement FPGA detection 
        # Intel/Altera and Xilinx
        return None
        
    def _get_total_memory(self) -> int:
        """Get total system memory in MB"""
        try:
            if self.platform["is_micropython"]:
                import gc
                return gc.mem_free() // (1024 * 1024)
            else:
                import psutil
                return psutil.virtual_memory().total // (1024 * 1024)
        except:
            return 0
            
    def _detect_cpu_model(self) -> str:
        """Detect CPU model"""
        if self.platform["os"] == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            except:
                pass
        return platform.processor() or "Unknown"
        
    def _detect_cpu_features(self) -> List[str]:
        """Detect CPU features like SIMD support"""
        features = []
        if self.platform["os"] == "Linux":
            try:
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
        
    def get_recommended_config(self) -> Dict:
        """Get recommended benchmark configuration based on detected hardware"""
        config = {
            "framework": "tflite",  # Default to TFLite
            "quantization": [],
            "batch_size": 1,
            "threads": 1
        }
        
        # Adjust based on detected capabilities
        if any(acc["type"] == "RKNN" for acc in self.capabilities["accelerators"]):
            config["framework"] = "rknn"
        elif any(acc["type"] == "CVITEK" for acc in self.capabilities["accelerators"]):
            config["framework"] = "cvitek"
        elif any(acc["type"] == "TINYML" for acc in self.capabilities["accelerators"]):
            config["framework"] = "tflite-micro"
            config["quantization"] = ["int8"]
            
        return config
        
    def print_capabilities(self):
        """Print detected capabilities in a readable format"""
        print("\nHardware Detection Results")
        print("=" * 40)
        print(f"Platform: {self.platform['os']} ({self.platform['architecture']})")
        print(f"Python: {self.platform['python_impl']}")
        print(f"Total Memory: {self.capabilities['memory_mb']} MB")
        
        print("\nAccelerators:")
        for acc in self.capabilities["accelerators"]:
            print(f"- Type: {acc['type']}")
            print(f"  Model: {acc.get('model', 'Unknown')}")
            if "ops" in acc:
                print(f"  Supported Ops: {', '.join(acc['ops'])}")
            if "features" in acc:
                print(f"  Features: {', '.join(acc['features'])}")
                
        if self.capabilities["quantization"]:
            print("\nSupported Quantization:")
            print(", ".join(self.capabilities["quantization"]))
            
def main():
    detector = HardwareDetector()
    detector.print_capabilities()
    
if __name__ == "__main__":
    main()
