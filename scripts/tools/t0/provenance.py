"""Exact pre/post identity, complete asset snapshot and frozen output hashes."""

from pathlib import Path
import shutil
from t0lib import REPO_ROOT, sha256_file, read_json, write_json, package_versions

PACKAGES = ["torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab", "rsl-rl-lib", "numpy", "scipy"]


def inputs(root=REPO_ROOT):
    paths = set()
    for folder in ("src", "scripts/tools/t0", "scripts/tools/morphology", "tests/t0", "tests/env"):
        paths.update(
            p.relative_to(root).as_posix()
            for p in (root / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in (".py", ".xml", ".urdf", ".STL", ".stl", ".npy")
        )
    paths.update(["pyproject.toml", "docs/research/morphology/evidence/library.json"])
    return {
        rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in sorted(paths)
    }


def backend_identity():
    import mjlab, mujoco_warp, rsl_rl

    modules = {}
    for package in (mjlab, mujoco_warp, rsl_rl):
        base = Path(package.__file__).parent
        modules.update({str(p): sha256_file(p) for p in sorted(base.rglob("*.py"))})
    return dict(versions=package_versions(PACKAGES), files=modules)


def freeze(run_dir):
    digest = inputs()
    if any(v is None for v in digest.values()):
        raise FileNotFoundError("mandatory input missing")
    snap = run_dir / "source_snapshot"
    snap.mkdir(exist_ok=False)
    for rel in digest:
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    write_json(
        run_dir / "input_identity.json",
        dict(schema=2, root=str(REPO_ROOT), files=digest, backend=backend_identity()),
    )


def verify_inputs(run_dir, root=REPO_ROOT):
    frozen = read_json(run_dir / "input_identity.json")
    current = inputs(root)
    issues = [
        rel
        for rel in sorted(set(current) | set(frozen["files"]))
        if current.get(rel) != frozen["files"].get(rel)
    ]
    issues += [
        "snapshot:" + rel
        for rel, digest in frozen["files"].items()
        if not (run_dir / "source_snapshot" / rel).is_file()
        or sha256_file(run_dir / "source_snapshot" / rel) != digest
    ]
    if backend_identity() != frozen["backend"]:
        issues.append("backend identity")
    return issues


def seal_outputs(run_dir):
    files = {
        p.relative_to(run_dir).as_posix(): sha256_file(p)
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
        and p.suffix in (".npz", ".json", ".mjb", ".pt")
        and "source_snapshot" not in p.parts
        and p.name not in ("output_identity.json", "acceptance.json")
        and "replay" not in p.parts
    }
    write_json(run_dir / "output_identity.json", dict(schema=2, files=files))


def verify_outputs(run_dir):
    frozen = read_json(run_dir / "output_identity.json")["files"]
    required = {
        "input_identity.json",
        "resolved_env_cfg.json",
        "resolved_agent_cfg.json",
        "hold_model.mjb",
        "hold_physics.npz",
        "hold_metadata.json",
        "hold_static.json",
        "episodes.npz",
        "eval_summary.json",
        "t0_short_model.pt",
    }
    issues = ["missing_manifest_entry:" + rel for rel in sorted(required - set(frozen))]
    issues += [
        rel
        for rel, digest in frozen.items()
        if not (run_dir / rel).is_file() or sha256_file(run_dir / rel) != digest
    ]
    return issues
