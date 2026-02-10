# Hardware Compatibility

## Tested Platforms

| Device | Architecture | Runtime | Status |
|--------|-------------|---------|--------|
| RP2040 (Raspberry Pi Pico) | ARM Cortex-M0+ | CircuitPython | ✅ Tested |
| RP2350 (Raspberry Pi Pico 2) | ARM Cortex-M33 | MicroPython, CircuitPython | ✅ Tested |
| ESP32-P4 | RISC-V | CircuitPython | ✅ Tested |
| Lyra Zero W | ARM (Linux) | CPython | ✅ Tested |

## Planned Testing

The following platforms are planned for testing before the next release:

| Device | Architecture | Runtime | Board Detection ID |
|--------|-------------|---------|-------------------|
| i7-14700K | x86-64 | CPython | `x86_64_linux` or `windows` |
| Luckfox Omni3576 (RK3576) | ARM64 | CPython | `luckfox_omni3576` |
| SpacemiT MUSE Pi Pro (K1) | RISC-V 64 | CPython | `spacemit_k1` |
| Luckfox Pico Zero (RV1106) | ARM32 | CPython | `luckfox_pico` |
| Milk-V Duo 256 (C906) | RISC-V 64 | CPython | `milkv_duo` |
| ESP32-S3 | Xtensa LX7 | MicroPython | `esp32s3` |
| ESP32-C6 | RISC-V 32 | MicroPython | `esp32c6` |
| ESP32 (D0WDR2) | Xtensa LX6 | MicroPython | `esp32` |
| RP2350 | ARM Cortex-M33 | MicroPython | `rp2350` |
| RP2040 | ARM Cortex-M0+ | CircuitPython | `rp2040` |
| RP2350 | ARM Cortex-M33 | CircuitPython | `rp2350` |

## Hardware Detection Methods

### CPython (Linux)

The Python benchmark detects hardware through:

- `/proc/cpuinfo` — CPU model, frequency, architecture flags
- `/sys/firmware/devicetree/base/model` or `/proc/device-tree/model` — SBC model identification
- `platform.machine()` — architecture (arm, aarch64, x86_64, riscv64)
- `psutil.cpu_freq()` — current CPU frequency
- `psutil.virtual_memory()` — available RAM
- `/sys/class/thermal/thermal_zone0/temp` — CPU temperature

### MicroPython

The MicroPython benchmark detects hardware through:

- `machine.freq()` — CPU frequency
- `sys.implementation._machine` — board identification string
- `machine.ADC(4)` — on-chip temperature sensor (RP2040/RP2350)
- `gc.mem_free()` — available heap memory

### CircuitPython

The CircuitPython benchmark detects hardware through:

- `microcontroller.cpu.frequency` — CPU frequency
- `microcontroller.cpus` — core count
- `microcontroller.cpu.temperature` — on-chip temperature
- `board.board_id` — board identification
- `gc.mem_free()` — available heap memory
