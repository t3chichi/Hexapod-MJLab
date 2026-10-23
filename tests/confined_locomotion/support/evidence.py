"""Small, functional evidence-writing helpers shared by current GPU checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_repo_root() -> Path:
  """Resolve and validate the repository root from this helper's real path."""
  root = Path(__file__).resolve().parents[3]
  if not (root / ".git").exists() or not (root / "src").is_dir():
    raise RuntimeError(f"could not resolve repository root from {__file__}: {root}")
  return root


def fresh_directory(path: str | Path) -> Path:
  """Create a new evidence directory; never reuse or overwrite a prior run."""
  target = Path(path).expanduser().resolve()
  target.parent.mkdir(parents=True, exist_ok=True)
  target.mkdir(parents=False, exist_ok=False)
  return target


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
  """Write a finite JSON record atomically, preserving a trailing newline."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
  rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)
  try:
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, target)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  return target


def write_manifest(path: str | Path, *, files: list[str | Path], recipe: dict) -> dict:
  """Record immutable output-file hashes next to the functional recipe."""
  entries = []
  for file_path in sorted((Path(p) for p in files), key=lambda p: str(p)):
    if not file_path.is_file():
      raise FileNotFoundError(file_path)
    entries.append({"path": str(file_path), "sha256": sha256_file(file_path), "bytes": file_path.stat().st_size})
  manifest = {"schema": "confined_locomotion.gpu_evidence/v1", "recipe": recipe, "files": entries}
  write_json(path, manifest)
  return manifest

