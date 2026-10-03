"""Read-only, hash-checked source imports; never initializes Isaac or old tasks."""

import os
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path


def source_root():
    return Path(os.environ.get("EA2_LEGACY_ROOT", str(Path.home() / "el4090_legged_gym")))


def load_reference():
    root = source_root()
    manifest = json.loads((Path(__file__).parents[1] / "data/ea2/source_manifest.json").read_text())
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    source = root / "legged_gym/legged_gym"
    prefix = "legged_gym.envs.el_4090.envelope_adaptive_2"
    for name in [
        "legged_gym",
        "legged_gym.envs",
        "legged_gym.envs.el_4090",
        prefix,
        "legged_gym.utils",
        "legged_gym.utils.envelop",
        "legged_gym.utils.envelop.network",
    ]:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    haa = load(
        "legged_gym.utils.envelop.network.haa_swing_range",
        source / "utils/envelop/network/haa_swing_range.py",
    )
    modules = {}
    for name in [
        "_contracts",
        "envelope_geometry",
        "envelope_oracle",
        "target_smoother",
        "path_planner",
        "map_generator",
        "path_batch",
        "airy_mount",
        "range_image",
    ]:
        modules[name] = load(
            prefix + "." + name, source / "envs/el_4090/envelope_adaptive_2" / f"{name}.py"
        )
    modules["haa"] = haa
    pkg = types.ModuleType(prefix + ".sl")
    pkg.__path__ = []
    sys.modules[prefix + ".sl"] = pkg
    for name in ["sl_config", "model", "dataset", "train"]:
        modules["sl_" + name] = load(
            prefix + ".sl." + name, source / "envs/el_4090/envelope_adaptive_2/sl" / f"{name}.py"
        )
    return modules
