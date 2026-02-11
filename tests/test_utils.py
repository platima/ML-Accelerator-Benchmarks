"""
Unit tests for the utils package.

Run with::

    python -m pytest tests/ -v
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from utils.benchmark_analyzer import BenchmarkAnalyser, BenchmarkResult, load_result
from utils.results_handler import ResultsHandler
from utils.validate_results import (
    _check_consistency,
    _check_required,
    _check_types,
    load_schema,
    validate_directory,
    validate_file,
)
from utils.hardware_detect import HardwareDetector
from utils.micropython_viz import TextVisualiser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESULT = {
    "_meta": {
        "Source version": "0.9.0",
        "Source code": "https://example.com/bench.py",
        "Source repo": "https://example.com/repo",
        "Test date": "2026-01-01",
        "Tester": "UnitTest",
        "Firmware": "test-firmware",
        "Notes": "",
    },
    "device": {
        "board_type": "test_board",
        "cpu_freq_mhz": 200.0,
        "num_cores": 2,
        "temp_sensor": True,
        "power_sensor": False,
    },
    "performance": {
        "channels": 3,
        "array_size": 100,
        "memory_total": 500000,
        "memory_used": 250000,
        "min_inference_ms": 90.0,
        "max_inference_ms": 110.0,
        "avg_inference_ms": 100.0,
        "throughput_fps": 10.0,
        "avg_temperature": 45.0,
        "max_temperature": 50.0,
    },
    "benchmark": {
        # total_ops = 100^3 * 3 + 100^2 * 3 * 2 = 3_000_000 + 60_000 = 3_060_000
        "total_ops": 3060000,
        # ops_per_second = 3_060_000 / (100 / 1000) = 30_600_000
        "ops_per_second": 30600000.0,
        # normalized_score = 30_600_000 / 200 = 153_000
        "normalized_score": 153000.0,
        # theoretical_power = 153_000 * 200 * 2 = 61_200_000
        "theoretical_power": 61200000.0,
    },
}


@pytest.fixture
def sample_result_file(tmp_path):
    """Write SAMPLE_RESULT to a temp JSON file and return its path."""
    fp = tmp_path / "2026-01-01 TestBoard MicroPython.json"
    fp.write_text(json.dumps(SAMPLE_RESULT, indent=2))
    return str(fp)


@pytest.fixture
def results_dir(tmp_path):
    """Create a temp directory with two result files."""
    for i, board in enumerate(["board_alpha", "board_beta"]):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        result["device"]["board_type"] = board
        result["device"]["cpu_freq_mhz"] = 100.0 * (i + 1)
        # Recalculate normalized_score for consistency
        result["benchmark"]["normalized_score"] = (
            result["benchmark"]["ops_per_second"] / result["device"]["cpu_freq_mhz"]
        )
        result["benchmark"]["theoretical_power"] = (
            result["benchmark"]["normalized_score"]
            * result["device"]["cpu_freq_mhz"]
            * result["device"]["num_cores"]
        )
        fp = tmp_path / "2026-01-0{} {} Test.json".format(i + 1, board)
        fp.write_text(json.dumps(result, indent=2))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# benchmark_analyzer tests
# ---------------------------------------------------------------------------


class TestLoadResult:
    def test_load_valid_file(self, sample_result_file):
        r = load_result(sample_result_file)
        assert isinstance(r, BenchmarkResult)
        assert r.board_type == "test_board"
        assert r.cpu_freq_mhz == 200.0
        assert r.num_cores == 2
        assert r.array_size == 100
        assert r.avg_inference_ms == 100.0
        assert r.ops_per_second == 30600000.0
        assert r.normalized_score == 153000.0
        assert r.avg_temperature == 45.0

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_result("/nonexistent/path.json")

    def test_load_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        with pytest.raises(json.JSONDecodeError):
            load_result(str(bad))

    def test_load_missing_optional_fields(self, tmp_path):
        """Result with no temperature should still load."""
        result = json.loads(json.dumps(SAMPLE_RESULT))
        del result["performance"]["avg_temperature"]
        del result["performance"]["max_temperature"]
        fp = tmp_path / "no_temp.json"
        fp.write_text(json.dumps(result))
        r = load_result(str(fp))
        assert r.avg_temperature is None
        assert r.max_temperature is None


class TestBenchmarkAnalyser:
    def test_load_from_directory(self, results_dir):
        analyser = BenchmarkAnalyser(results_dir=results_dir)
        assert len(analyser.results) == 2

    def test_sorted_by_default(self, results_dir):
        analyser = BenchmarkAnalyser(results_dir=results_dir)
        ranked = analyser.sorted_by()
        # Higher normalized_score first
        assert ranked[0].normalized_score >= ranked[1].normalized_score

    def test_sorted_by_ascending(self, results_dir):
        analyser = BenchmarkAnalyser(results_dir=results_dir)
        ranked = analyser.sorted_by("cpu_freq_mhz", reverse=False)
        assert ranked[0].cpu_freq_mhz <= ranked[1].cpu_freq_mhz

    def test_generate_comparison_table(self, results_dir):
        analyser = BenchmarkAnalyser(results_dir=results_dir)
        table = analyser.generate_comparison_table()
        assert "Board" in table
        assert "board_alpha" in table
        assert "board_beta" in table

    def test_generate_report(self, results_dir):
        analyser = BenchmarkAnalyser(results_dir=results_dir)
        report = analyser.generate_report()
        assert "ML Accelerator Benchmark Analysis Report" in report
        assert "board_alpha" in report

    def test_skip_schema_file(self, results_dir):
        """Should not try to load results-schema.json."""
        schema = Path(results_dir) / "results-schema.json"
        schema.write_text(json.dumps({"$schema": "test"}))
        analyser = BenchmarkAnalyser(results_dir=results_dir)
        assert len(analyser.results) == 2  # not 3

    def test_explicit_files(self, sample_result_file):
        analyser = BenchmarkAnalyser(result_files=[sample_result_file])
        assert len(analyser.results) == 1
        assert analyser.results[0].board_type == "test_board"


# ---------------------------------------------------------------------------
# validate_results tests
# ---------------------------------------------------------------------------


class TestValidator:
    def test_validate_valid_file(self, sample_result_file):
        schema = load_schema(
            os.path.join("results", "results-schema.json")
        )
        ok, errors = validate_file(sample_result_file, schema)
        assert ok, "Expected valid file, got errors: {}".format(errors)

    def test_validate_missing_section(self, tmp_path):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        del result["benchmark"]
        fp = tmp_path / "missing_bench.json"
        fp.write_text(json.dumps(result))
        schema = load_schema()
        ok, errors = validate_file(str(fp), schema)
        assert not ok
        assert any("benchmark" in e for e in errors)

    def test_validate_missing_meta(self, tmp_path):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        del result["_meta"]
        fp = tmp_path / "missing_meta.json"
        fp.write_text(json.dumps(result))
        schema = load_schema()
        ok, errors = validate_file(str(fp), schema)
        assert not ok
        assert any("_meta" in e for e in errors)

    def test_validate_bad_type(self, tmp_path):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        result["device"]["num_cores"] = "two"  # should be integer
        fp = tmp_path / "bad_type.json"
        fp.write_text(json.dumps(result))
        schema = load_schema()
        ok, errors = validate_file(str(fp), schema)
        assert not ok
        assert any("num_cores" in e for e in errors)

    def test_validate_directory(self, results_dir):
        schema_path = os.path.join("results", "results-schema.json")
        passed, failed, failures = validate_directory(
            results_dir, schema_path
        )
        assert passed == 2
        assert failed == 0

    def test_consistency_correct(self):
        errors = _check_consistency(SAMPLE_RESULT)
        assert errors == [], "Expected no consistency errors, got: {}".format(errors)

    def test_consistency_wrong_total_ops(self):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        result["benchmark"]["total_ops"] = 999  # way off
        errors = _check_consistency(result)
        assert any("total_ops" in e for e in errors)

    def test_bool_not_accepted_as_integer(self):
        """bool should not pass an integer type check."""
        errors = _check_types(
            {"num_cores": True},
            {"num_cores": {"type": "integer"}},
            "device",
        )
        assert len(errors) == 1
        assert "boolean" in errors[0]

    def test_bool_not_accepted_as_number(self):
        errors = _check_types(
            {"cpu_freq_mhz": False},
            {"cpu_freq_mhz": {"type": "number"}},
            "device",
        )
        assert len(errors) == 1
        assert "boolean" in errors[0]


# ---------------------------------------------------------------------------
# results_handler tests
# ---------------------------------------------------------------------------


class TestResultsHandler:
    def test_save_and_load(self, tmp_path):
        handler = ResultsHandler(results_dir=str(tmp_path))
        filepath = handler.save_result(SAMPLE_RESULT, "TestDev")
        assert filepath != ""
        assert os.path.exists(filepath)

        loaded = handler.load_result(os.path.basename(filepath))
        assert loaded["device"]["board_type"] == "test_board"

    def test_list_results(self, results_dir):
        handler = ResultsHandler(results_dir=results_dir)
        listing = handler.list_results()
        assert len(listing) == 2

    def test_list_results_filter(self, results_dir):
        handler = ResultsHandler(results_dir=results_dir)
        listing = handler.list_results("board_alpha")
        assert len(listing) == 1

    def test_get_latest(self, results_dir):
        handler = ResultsHandler(results_dir=results_dir)
        latest = handler.get_latest_result()
        assert latest != {}

    def test_compare_results(self, results_dir):
        handler = ResultsHandler(results_dir=results_dir)
        files = handler.list_results()
        compared = handler.compare_results(files)
        assert len(compared) == 2
        assert "_filename" in compared[0]

    def test_load_nonexistent(self, tmp_path):
        handler = ResultsHandler(results_dir=str(tmp_path))
        result = handler.load_result("does_not_exist.json")
        assert result == {}

    def test_skip_schema(self, results_dir):
        """Schema file should not appear in list_results."""
        schema = Path(results_dir) / "results-schema.json"
        schema.write_text("{}")
        handler = ResultsHandler(results_dir=results_dir)
        listing = handler.list_results()
        assert "results-schema.json" not in listing


# ---------------------------------------------------------------------------
# hardware_detect tests
# ---------------------------------------------------------------------------


class TestHardwareDetector:
    def test_instantiates(self):
        detector = HardwareDetector()
        assert detector.platform_info["python_impl"] in (
            "CPython",
            "MicroPython",
            "CircuitPython",
        )

    def test_summary_string(self):
        detector = HardwareDetector()
        summary = detector.summary()
        assert "Hardware Detection Summary" in summary
        assert "OS:" in summary

    def test_cpu_info_populated(self):
        detector = HardwareDetector()
        assert "model" in detector.cpu_info
        assert "cores" in detector.cpu_info
        assert detector.cpu_info["cores"] >= 1

    def test_accelerators_list(self):
        detector = HardwareDetector()
        assert isinstance(detector.accelerators, list)
        assert len(detector.accelerators) >= 1
        assert detector.accelerators[0]["type"] == "CPU"


# ---------------------------------------------------------------------------
# micropython_viz tests
# ---------------------------------------------------------------------------


class TestTextVisualiser:
    def test_fmt_number_large(self):
        assert TextVisualiser.fmt_number(1_500_000) == "1.5M"
        assert TextVisualiser.fmt_number(2_300) == "2.3K"
        assert TextVisualiser.fmt_number(42) == "42.0"
        assert TextVisualiser.fmt_number(3_000_000_000) == "3.0G"

    def test_bar_full(self):
        vis = TextVisualiser(width=10)
        bar = vis.bar(100, 100)
        assert bar == "[##########]"

    def test_bar_empty(self):
        vis = TextVisualiser(width=10)
        bar = vis.bar(0, 100)
        assert bar == "[----------]"

    def test_bar_half(self):
        vis = TextVisualiser(width=10)
        bar = vis.bar(50, 100)
        assert bar == "[#####-----]"

    def test_bar_zero_max(self):
        vis = TextVisualiser(width=10)
        bar = vis.bar(50, 0)
        assert bar == "[----------]"

    def test_show_result(self, capsys):
        vis = TextVisualiser()
        vis.show_result(SAMPLE_RESULT)
        captured = capsys.readouterr()
        assert "test_board" in captured.out
        assert "153000" in captured.out or "153000.0" in captured.out

    def test_show_comparison(self, capsys):
        vis = TextVisualiser()
        vis.show_comparison([SAMPLE_RESULT, SAMPLE_RESULT])
        captured = capsys.readouterr()
        assert "Score Comparison" in captured.out

    def test_show_summary_table(self, capsys):
        vis = TextVisualiser()
        vis.show_summary_table([SAMPLE_RESULT])
        captured = capsys.readouterr()
        assert "test_board" in captured.out


# ---------------------------------------------------------------------------
# Schema integration — validate the real results/ directory
# ---------------------------------------------------------------------------


class TestRealResults:
    """Validate the actual result files checked into the repo."""

    def test_all_results_pass_validation(self):
        schema = load_schema()
        passed, failed, failures = validate_directory("results")
        failure_msgs = []
        for name, errs in failures:
            failure_msgs.append("{}: {}".format(name, "; ".join(errs)))
        assert failed == 0, "Failures:\n" + "\n".join(failure_msgs)
        assert passed >= 1
