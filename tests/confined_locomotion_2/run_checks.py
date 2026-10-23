"""Dependency-free CPU test entry (also usable when pytest is unavailable)."""

import importlib.util
import inspect
from pathlib import Path
import tempfile


def main():
    passed = 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location("v2_checks_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in vars(module).items():
            if name.startswith("test_") and inspect.isfunction(function):
                with tempfile.TemporaryDirectory(prefix="confined-v2-test-") as temporary:
                    kwargs = {name: Path(temporary) for name in inspect.signature(function).parameters
                              if name == "tmp_path"}
                    function(**kwargs)
                passed += 1
                print("PASS", path.name, name)
    print(f"{passed} CPU checks passed; no GPU or learning-effect claim")


if __name__ == "__main__":
    main()
