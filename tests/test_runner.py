"""Failure propagation, input validation and immutable regression outputs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run as runner


class RunnerTests(unittest.TestCase):
    def test_empty_duplicate_missing_selections(self):
        check = runner.Check("same", "tests/test_runner.py")
        for selection in ((), (check, check)):
            with self.assertRaises(ValueError):
                runner.validate_checks(selection)
        with self.assertRaises(FileNotFoundError):
            runner.validate_checks((runner.Check("missing", "missing_test.py"),))

    def test_failure_does_not_hide_remaining_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "failure.py").write_text("raise SystemExit(7)\n")
            (root / "success.py").write_text("print('complete')\n")
            output = root / "output"
            checks = (runner.Check("failure", "failure.py"), runner.Check("success", "success.py"))
            with patch.object(runner, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_checks(checks, output), 1)
            report = json.loads((output / "summary.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual([row["returncode"] for row in report["checks"]], [7, 0])
            self.assertIn("complete", (output / "success.log").read_text())
            with patch.object(runner, "ROOT", root), self.assertRaises(FileExistsError):
                runner.run_checks(checks, output)
            self.assertEqual(report, json.loads((output / "summary.json").read_text()))

    def test_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slow.py").write_text("pass\n")
            error = runner.subprocess.TimeoutExpired(["python"], 1)
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner.subprocess, "run", side_effect=error),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    runner.run_checks((runner.Check("slow", "slow.py"),), root / "out"), 1
                )
            row = json.loads((root / "out/summary.json").read_text())["checks"][0]
            self.assertEqual((row["state"], row["returncode"]), ("FAIL", 124))

    def test_profiles_have_distinct_existing_checks(self):
        for profile in runner.PROFILES.values():
            runner.validate_checks(profile)
        cpu = {check.name for check in runner.CPU}
        self.assertNotIn("ppo_learning", cpu)
        self.assertNotIn("envelope_source_parity", cpu)


if __name__ == "__main__":
    unittest.main(verbosity=2)
