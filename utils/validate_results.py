#!/usr/bin/env python3
"""
Result Schema Validator

Validates benchmark result JSON files against the project schema
(``results/results-schema.json``).  Can be run standalone or imported
as a module.

Usage::

    python -m utils.validate_results                    # validate all
    python -m utils.validate_results results/2025*.json # specific files
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple


def load_schema(schema_path: str = None) -> dict:
    """Load the JSON schema file.

    Args:
        schema_path: Explicit path. Falls back to
            ``results/results-schema.json``.

    Returns:
        Parsed schema dictionary.
    """
    if schema_path is None:
        schema_path = os.path.join("results", "results-schema.json")
    with open(schema_path, "r") as fh:
        return json.load(fh)


def _check_required(data: dict, required: list, path: str) -> List[str]:
    """Check that all *required* keys exist in *data*."""
    errors = []
    for key in required:
        if key not in data:
            errors.append("Missing required key '{}' in {}".format(key, path))
    return errors


def _check_types(data: dict, properties: dict, path: str) -> List[str]:
    """Check value types match the schema ``type`` declarations."""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    errors = []
    for key, spec in properties.items():
        if key not in data:
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        py_type = type_map.get(expected)
        if py_type is None:
            continue
        val = data[key]
        # In Python, bool is a subclass of int — guard against that
        if expected == "integer" and isinstance(val, bool):
            errors.append(
                "{}.{}: expected integer, got boolean".format(path, key)
            )
            continue
        if expected == "number" and isinstance(val, bool):
            errors.append(
                "{}.{}: expected number, got boolean".format(path, key)
            )
            continue
        # JSON integer / number flexibility
        if expected == "integer" and isinstance(val, float) and val == int(val):
            continue
        if not isinstance(val, py_type):
            errors.append(
                "{}.{}: expected {}, got {}".format(
                    path, key, expected, type(val).__name__
                )
            )
    return errors


def _check_consistency(data: dict) -> List[str]:
    """Verify computed benchmark fields are internally consistent.

    Allows a 5 % tolerance to accommodate floating-point rounding in
    firmware that only has single-precision floats.
    """
    errors = []
    perf = data.get("performance", {})
    bench = data.get("benchmark", {})
    device = data.get("device", {})

    array_size = perf.get("array_size", 0)
    channels = perf.get("channels", 3)
    avg_ms = perf.get("avg_inference_ms", 0)
    freq = device.get("cpu_freq_mhz", 0)
    cores = device.get("num_cores", 1)

    # total_ops = array_size^3 * channels + array_size^2 * channels * 2
    expected_ops = (array_size ** 3) * channels + (array_size ** 2) * channels * 2
    actual_ops = bench.get("total_ops", 0)
    if expected_ops and actual_ops:
        if not math.isclose(expected_ops, actual_ops, rel_tol=0.05):
            errors.append(
                "total_ops mismatch: expected ~{}, got {}".format(
                    expected_ops, actual_ops
                )
            )

    # ops_per_second = total_ops / (avg_inference_ms / 1000)
    if avg_ms > 0 and actual_ops:
        expected_ops_s = actual_ops / (avg_ms / 1000.0)
        actual_ops_s = bench.get("ops_per_second", 0)
        if actual_ops_s and not math.isclose(expected_ops_s, actual_ops_s, rel_tol=0.05):
            errors.append(
                "ops_per_second mismatch: expected ~{:.0f}, got {:.0f}".format(
                    expected_ops_s, actual_ops_s
                )
            )

    # normalized_score = ops_per_second / cpu_freq_mhz
    ops_s = bench.get("ops_per_second", 0)
    if freq > 0 and ops_s:
        expected_norm = ops_s / freq
        actual_norm = bench.get("normalized_score", 0)
        if actual_norm and not math.isclose(expected_norm, actual_norm, rel_tol=0.05):
            errors.append(
                "normalized_score mismatch: expected ~{:.1f}, got {:.1f}".format(
                    expected_norm, actual_norm
                )
            )

    # theoretical_power = normalized_score * cpu_freq_mhz * num_cores
    norm = bench.get("normalized_score", 0)
    if norm and freq and cores:
        expected_power = norm * freq * cores
        actual_power = bench.get("theoretical_power", 0)
        if actual_power and not math.isclose(expected_power, actual_power, rel_tol=0.05):
            errors.append(
                "theoretical_power mismatch: expected ~{:.0f}, got {:.0f}".format(
                    expected_power, actual_power
                )
            )

    return errors


def validate_file(filepath: str, schema: dict) -> Tuple[bool, List[str]]:
    """Validate a single result JSON file.

    Args:
        filepath: Path to the ``.json`` file.
        schema: Parsed JSON schema dictionary.

    Returns:
        A ``(valid, errors)`` tuple.
    """
    errors: List[str] = []

    try:
        with open(filepath, "r") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return False, ["Invalid JSON: {}".format(exc)]

    if not isinstance(data, dict):
        return False, ["Root element must be an object"]

    # Top-level required sections
    for section in schema.get("required", []):
        if section not in data:
            errors.append("Missing required section '{}'".format(section))

    # Per-section checks
    for section_name, section_schema in schema.get("properties", {}).items():
        if section_name not in data:
            continue
        section_data = data[section_name]
        if not isinstance(section_data, dict):
            errors.append("'{}' must be an object".format(section_name))
            continue

        required = section_schema.get("required", [])
        errors.extend(_check_required(section_data, required, section_name))

        props = section_schema.get("properties", {})
        errors.extend(_check_types(section_data, props, section_name))

    # Internal consistency
    errors.extend(_check_consistency(data))

    return len(errors) == 0, errors


def validate_directory(
    results_dir: str = "results",
    schema_path: str = None,
) -> Tuple[int, int, List[Tuple[str, List[str]]]]:
    """Validate all JSON result files in a directory.

    Args:
        results_dir: Directory containing ``.json`` result files.
        schema_path: Path to the schema file.

    Returns:
        ``(passed, failed, failures)`` where *failures* is a list of
        ``(filename, errors)`` tuples.
    """
    schema = load_schema(schema_path)
    passed = 0
    failed = 0
    failures: List[Tuple[str, List[str]]] = []

    results_path = Path(results_dir)
    for fp in sorted(results_path.glob("*.json")):
        if fp.name == "results-schema.json":
            continue
        ok, errs = validate_file(str(fp), schema)
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((fp.name, errs))

    return passed, failed, failures


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate benchmark result JSON files against the schema"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific result files to validate (default: all in results/)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to the JSON schema (default: results/results-schema.json)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to scan when no files are specified",
    )
    args = parser.parse_args()

    schema = load_schema(args.schema)

    if args.files:
        total_pass = 0
        total_fail = 0
        for fp in args.files:
            ok, errs = validate_file(fp, schema)
            basename = os.path.basename(fp)
            if ok:
                print("  PASS  {}".format(basename))
                total_pass += 1
            else:
                print("  FAIL  {}".format(basename))
                for e in errs:
                    print("        - {}".format(e))
                total_fail += 1
    else:
        total_pass, total_fail, failures = validate_directory(
            args.results_dir, args.schema
        )
        for name, errs in failures:
            print("  FAIL  {}".format(name))
            for e in errs:
                print("        - {}".format(e))

        passed_files = total_pass
        for fp in sorted(Path(args.results_dir).glob("*.json")):
            if fp.name == "results-schema.json":
                continue
            if not any(fp.name == f[0] for f in failures):
                print("  PASS  {}".format(fp.name))

    print("\n{} passed, {} failed".format(total_pass, total_fail))
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
