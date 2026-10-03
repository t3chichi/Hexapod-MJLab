"""T0-A shared library: run directories, provenance, hashing, verdicts.

Every T0-A entry point uses this module so that all artifacts land in one
exclusive ``logs/t0/<run_id>/`` directory with identical provenance rules:

* the run directory is created once and never overwritten (existing path is
  rejected);
* every stage writes ``<stage>.json`` with ``argv``, ``returncode``,
  ``started/finished`` timestamps and per-gate verdicts;
* a required gate that is FAIL or NOT_VERIFIED makes the process exit
  non-zero (callers must not swallow it with ``|| true``);
* source/model/input/checkpoint identities are SHA-256 frozen.

This module is tooling only: it never modifies mjlab, the shared RL stack or
robot assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PASS = "PASS"
FAIL = "FAIL"
NOT_VERIFIED = "NOT_VERIFIED"
PARTIAL = "PARTIAL"
BLOCKED = "BLOCKED"

_STATES = {PASS, FAIL, NOT_VERIFIED, PARTIAL, BLOCKED}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_run_dir(base: Path | None = None, prefix: str = "t0a") -> Path:
    """Create an exclusive new run directory ``logs/t0/<prefix>-<timestamp>``."""
    base = base or (REPO_ROOT / "logs" / "t0")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{prefix}-{ts}"
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def reject_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def _json_default(obj):
    """Convert torch/numpy scalars-arrays to plain Python for JSON."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_json(path: Path, obj) -> None:
    # Validate before opening; every scientific artifact is immutable.
    payload = json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(payload + "\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class GateRecord:
    """One scientific gate with independent fields."""

    name: str
    state: str
    evidence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "evidence": self.evidence,
            "notes": self.notes,
        }


class StageReport:
    """Accumulates gates for one stage and enforces non-zero exit on failure."""

    def __init__(self, stage: str, run_dir: Path, argv=None):
        self.stage = stage
        self.run_dir = run_dir
        self.started = utc_now_iso()
        self.t0 = time.monotonic()
        self.argv = list(argv if argv is not None else sys.argv)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ("." + stage + ".claim")).touch(exist_ok=False)
        if (run_dir / (stage + ".json")).exists():
            raise FileExistsError(stage)
        self.gates: list[GateRecord] = []
        self.meta: dict = {}

    def gate(self, name: str, state: str, **evidence) -> GateRecord:
        if state not in _STATES:
            raise ValueError(f"unknown gate state {state}")
        if any(g.name == name for g in self.gates):
            raise ValueError(f"duplicate gate {name}")
        rec = GateRecord(name=name, state=state, evidence=evidence)
        self.gates.append(rec)
        print(
            f"[{self.stage}] gate {name}: {state}" + (f" {evidence}" if evidence else ""),
            flush=True,
        )
        return rec

    def note(self, msg: str) -> None:
        print(f"[{self.stage}] note: {msg}", flush=True)

    def finalize(self, extra_meta: dict | None = None, returncode: int | None = None) -> dict:
        required_failed = [g for g in self.gates if g.state != PASS]
        passed = bool(self.gates) and not required_failed
        if returncode is None or (returncode == 0 and not passed):
            returncode = 0 if passed else 1
        doc = {
            "stage": self.stage,
            "argv": self.argv,
            "started": self.started,
            "finished": utc_now_iso(),
            "wall_time_s": round(time.monotonic() - self.t0, 3),
            "returncode": returncode,
            "pipeline_passed": passed and returncode == 0,
            "gates": [g.to_dict() for g in self.gates],
            "meta": {**self.meta, **(extra_meta or {})},
        }
        out = self.run_dir / f"{self.stage}.json"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite stage report: {out}")
        write_json(out, doc)
        print(f"[{self.stage}] returncode={returncode} -> {out}", flush=True)
        return doc


def python_identity() -> dict:
    import torch

    cuda = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        cuda["cuda_device"] = torch.cuda.get_device_name(0)
        cuda["cuda_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 4)
    return {
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda": cuda,
    }


def package_versions(packages: list[str]) -> dict:
    import importlib.metadata as md

    out = {}
    for p in packages:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def gpu_smi_free_mb() -> float | None:
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            .stdout.strip()
            .splitlines()
        )
        return float(out[0])
    except Exception:
        return None


def git_state(repo: Path | None = None) -> dict:
    repo = repo or REPO_ROOT

    def run(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        ).stdout.strip()

    # -z: NUL-separated, no quoting/escaping of non-ASCII paths
    entries = []
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--short", "-z"],
        capture_output=True,
        text=True,
    ).stdout
    for token in out.split("\0"):
        if not token:
            continue
        entries.append(token)

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_entries": entries,
        "dirty_files": [e[3:] for e in entries],
    }
