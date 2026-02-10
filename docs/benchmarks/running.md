# Running Benchmarks

## CPython (python-benchmark.py)

For Linux SBCs and desktop systems with CPython and NumPy.

### Prerequisites

- Python 3.9+
- NumPy, psutil (install via `pip install -r requirements.txt`)

### Running

```bash
python benchmarks/python-benchmark.py
```

The script will:

1. Detect your CPU frequency, core count, and temperature sensor
2. Probe memory to find the maximum matrix size (up to 2048×2048)
3. Apply a 0.9× safety factor to the maximum size
4. Run 2 warmup iterations followed by 10 timed iterations
5. Print JSON results to stdout

### Customisation

Edit the `__main__` block at the bottom of the script to change:

- `channels` — number of matrix channels (default: 3)
- `warmup_runs` — number of warmup iterations (default: 2)
- `num_runs` — number of timed iterations (default: 10)

## MicroPython (micropython-benchmark.py)

For microcontrollers running MicroPython with ulab.

### Prerequisites

- MicroPython firmware with ulab support (e.g. [Pimoroni builds](https://github.com/pimoroni/pimoroni-pico-rp2350/releases))

### Running

Upload the script to your device and execute via REPL:

```python
exec(open('micropython-benchmark.py').read())
```

Or use Thonny / mpremote to run directly.

### Notes

- Maximum matrix size is capped at 256×256 for microcontrollers
- Memory is much more constrained — expect smaller matrices
- Timing uses `time.ticks_ms()` for millisecond precision

## CircuitPython (circuitpython-benchmark.py)

For microcontrollers running CircuitPython with ulab.

### Prerequisites

- CircuitPython 8.2+ firmware (includes ulab)

### Running

Copy the script to your device as `code.py` for auto-run, or execute via the REPL.

### Notes

- Maximum matrix size is capped at 256×256
- Timing uses `time.monotonic_ns()` for nanosecond precision
- A 0.9× safety factor is applied to the maximum matrix size

## Saving Results

All benchmarks print JSON to stdout. To save results, redirect output or copy the JSON block into a file in the `results/` directory. Use the naming convention:

```
YYYY-MM-DD Board Runtime.json
```

For example: `2025-01-19 RP2350 MicroPython.json`
