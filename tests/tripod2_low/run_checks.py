"""Run focused CPU contracts with the project environment, without pytest."""
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    passed = 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location("tripod_check_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        count = suite.countTestCases()
        if count:
            result = unittest.TextTestRunner(verbosity=1).run(suite)
            if not result.wasSuccessful():
                raise SystemExit(1)
            passed += count
        for name, function in vars(module).items():
            if name.startswith("test_") and inspect.isfunction(function):
                function()
                print("PASS", path.name, name, flush=True)
                passed += 1
    print(f"{passed} CPU checks passed. GPU runtime/training acceptance remains separate.")


if __name__ == "__main__":
    main()
