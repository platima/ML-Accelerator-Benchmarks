# Testing Checklist

This page lists every target device and the steps needed to verify the v0.9+ benchmark scripts.

## Test Matrix

| # | Device | Runtime | Script | Status |
|---|--------|---------|--------|--------|
| 1 | Intel i7-14700K (x86-64) | CPython + NumPy | `python-benchmark.py` | ⬜ |
| 2 | Luckfox Omni3576 (ARM64) | CPython + NumPy | `python-benchmark.py` | ⬜ |
| 3 | SpacemiT MUSE Pi Pro (RV64) | CPython + NumPy | `python-benchmark.py` | ⬜ |
| 4 | Luckfox Pico Zero (ARM32) | CPython + NumPy | `python-benchmark.py` | ⬜ |
| 5 | Milk-V Duo 256 (RV64) | CPython + NumPy | `python-benchmark.py` | ⬜ |
| 6 | RP2350 | MicroPython + ulab | `micropython-benchmark.py` | ⬜ |
| 7 | ESP32-S3 (LX7) | MicroPython + ulab | `micropython-benchmark.py` | ⬜ |
| 8 | ESP32-C6 (RV32) | MicroPython + ulab | `micropython-benchmark.py` | ⬜ |
| 9 | ESP32-D0WDR2 (LX6) | MicroPython + ulab | `micropython-benchmark.py` | ⬜ |
| 10 | RP2350 | CircuitPython + ulab | `circuitpython-benchmark.py` | ⬜ |
| 11 | RP2040 | CircuitPython + ulab | `circuitpython-benchmark.py` | ⬜ |

## Per-Device Steps

For each device:

1. **Upload** the appropriate benchmark script.
2. **Run** the benchmark and capture the JSON output.
3. **Save** the JSON to `results/YYYY-MM-DD <Board> <Runtime>.json`.
4. **Validate** with:
   ```bash
   python -m utils validate results/<filename>.json
   ```
5. **Check** the report:
   ```bash
   python -m utils analyse
   ```

## What to Look For

- [ ] No crashes or unhandled exceptions during the benchmark run.
- [ ] `board_type` is a recognisable identifier (not `unknown`).
- [ ] `cpu_freq_mhz` is non-zero and reasonable for the device.
- [ ] `num_cores` matches the device (e.g. 2 for RP2350, 1 for ESP32-C6).
- [ ] `array_size` is greater than zero.
- [ ] `avg_inference_ms` is positive and in a plausible range.
- [ ] `normalized_score` is positive.
- [ ] Validator reports `PASS` with no consistency errors.
- [ ] Temperature fields are present when the device has a sensor.

## After All Devices

Once all 11 devices have been tested:

```bash
# Validate every result file
python -m utils validate

# Generate the full comparison report
python -m utils analyse

# Generate charts (requires matplotlib)
python -m utils visualise
```
