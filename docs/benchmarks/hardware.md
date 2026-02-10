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

| Device | Architecture | Runtime | Notes |
|--------|-------------|---------|-------|
| i7-14700K | x86-64 | CPython | Desktop baseline |
| Luckfox Omni3576 (RK3576) | ARM64 | CPython | Debian Linux |
| SpacemiT MUSE Pi Pro (K1) | RISC-V 64 | CPython | Ubuntu Linux |
| Luckfox Pico Zero (RV1106) | ARM32 | CPython | Buildroot Linux |
| Milk-V Duo 256 (C906) | RISC-V 64 | CPython | Buildroot Linux |
| ESP32-S3 | Xtensa LX7 | MicroPython | Dual-core |
| ESP32-C6 | RISC-V 32 | MicroPython | Single-core |
| ESP32 (D0WDR2) | Xtensa LX6 | MicroPython | Dual-core, legacy |

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
