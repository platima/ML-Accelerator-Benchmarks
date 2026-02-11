# Understanding Results

## JSON Structure

Every benchmark produces a JSON object with four top-level sections:

```json
{
    "_meta": { ... },
    "device": { ... },
    "performance": { ... },
    "benchmark": { ... }
}
```

## Sections

### `_meta` — Metadata

| Field | Description |
|-------|-------------|
| `Source version` | Benchmark script version (semver) |
| `Source code` | URL to the exact source code used |
| `Source repo` | Repository URL |
| `Test date` | ISO 8601 date (YYYY-MM-DD) |
| `Tester` | Who ran the benchmark |
| `Firmware` | Firmware/runtime version or URL |
| `Notes` | Any relevant notes about the test run |

### `device` — Hardware Information

| Field | Type | Description |
|-------|------|-------------|
| `board_type` | string | Detected board identifier (e.g. `rp2350`, `arm_linux`) |
| `cpu_freq_mhz` | number | CPU frequency in MHz |
| `num_cores` | integer | Number of CPU cores detected |
| `temp_sensor` | boolean | Whether a temperature sensor is available |
| `power_sensor` | boolean | Whether a power/voltage sensor is available |

### `performance` — Timing and Resource Metrics

| Field | Type | Description |
|-------|------|-------------|
| `channels` | integer | Number of matrix channels used (default: 3) |
| `array_size` | integer | Matrix dimension N (NxN matrices) |
| `memory_total` | number | Total available memory in bytes at start |
| `memory_used` | number | Peak memory consumption in bytes during benchmark |
| `min_inference_ms` | number | Fastest iteration time in milliseconds |
| `max_inference_ms` | number | Slowest iteration time in milliseconds |
| `avg_inference_ms` | number | Mean iteration time in milliseconds |
| `throughput_fps` | number | Iterations per second (1000 / avg_inference_ms) |
| `avg_temperature` | number | *(Optional)* Mean CPU temperature in °C |
| `max_temperature` | number | *(Optional)* Peak CPU temperature in °C |

### `benchmark` — Derived Metrics

| Field | Type | Formula | Description |
|-------|------|---------|-------------|
| `total_ops` | number | N³×C + N²×C×2 | Total floating-point operations per iteration |
| `ops_per_second` | number | total_ops / (avg_ms / 1000) | Raw throughput |
| `normalized_score` | number | ops_per_second / cpu_freq_mhz | Frequency-independent efficiency score |
| `theoretical_power` | number | normalized_score × cpu_freq_mhz × num_cores | Multi-core adjusted throughput |

Where:

- **N** = `array_size`
- **C** = `channels`
- The `× 2` in `total_ops` accounts for the scaling and addition element-wise operations

## Schema Validation

Result files can be validated against the JSON schema at `results/results-schema.json`. The `avg_temperature` and `max_temperature` fields are optional — they are only present when the device has a working temperature sensor.

### Running the Validator

```bash
# Validate all result files in results/
python -m utils validate

# Validate specific files
python -m utils validate results/2025-01-19*.json
```

The validator checks:

1. **Structure** — all required sections and fields are present.
2. **Types** — values match the expected JSON types.
3. **Consistency** — derived fields (`total_ops`, `ops_per_second`, `normalized_score`, `theoretical_power`) agree with the formulas above within a 5 % tolerance.

A CI workflow runs the validator automatically whenever result files or the schema are modified.
