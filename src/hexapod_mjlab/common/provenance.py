"""Read-only Git provenance for local training recipes.

Only paths reported by Git as changed and not ignored are inspected. File
contents are represented by hashes; diff text is never collected or returned.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess


_SCHEMA = "hexapod_mjlab.code_provenance/v1"


def _git(root: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "core.refreshIndex=false", *args],
        stderr=subprocess.DEVNULL,
        env=env,
        timeout=10,
    )


def _worktree_digest(path: Path) -> str | None:
    try:
        if path.is_symlink():
            payload = os.fsencode(os.readlink(path))
        elif path.is_file():
            payload = path.read_bytes()
        else:
            return None
    except OSError:
        return None
    return hashlib.sha256(payload).hexdigest()


def _changed_paths(root: Path) -> list[tuple[str, str]]:
    raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames")
    changes = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        # With --no-renames each NUL-delimited record is XY<space>path.
        if len(record) < 4 or record[2:3] != b" ":
            continue
        changes.append((os.fsdecode(record[3:]), record[:2].decode("ascii", "replace")))
    return changes


def capture_code_provenance(source_path: str | os.PathLike[str] | None = None) -> dict:
    """Return a JSON-safe, read-only snapshot of this checkout's code state.

    Git absence or a source file outside a checkout is represented explicitly
    and never prevents a training run from starting.
    """
    source = Path(source_path or __file__).resolve()
    if shutil.which("git") is None:
        return {
            "schema": _SCHEMA,
            "status": "git_unavailable",
            "commit": None,
            "dirty": None,
            "changed_files": None,
            "content_sha256": None,
        }
    try:
        root = Path(_git(source.parent, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return {
            "schema": _SCHEMA,
            "status": "not_git_checkout",
            "commit": None,
            "dirty": None,
            "changed_files": None,
            "content_sha256": None,
        }
    try:
        commit = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    except subprocess.CalledProcessError:
        # An initialized repository can legitimately have no commit yet.
        commit = None
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return {
            "schema": _SCHEMA,
            "status": "git_unavailable",
            "commit": None,
            "dirty": None,
            "changed_files": None,
            "content_sha256": None,
        }
    try:
        changes = _changed_paths(root)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return {
            "schema": _SCHEMA,
            "status": "git_unavailable",
            "commit": commit,
            "dirty": None,
            "changed_files": None,
            "content_sha256": None,
        }

    changed_files = []
    for name, status in changes:
        path = root / name
        # `git rev-parse :path` reads the existing index only; it does not
        # refresh or otherwise write the index.
        try:
            index_blob = _git(root, "rev-parse", "--verify", f":{name}").decode("ascii").strip()
        except (OSError, subprocess.SubprocessError, UnicodeError):
            index_blob = None
        worktree_sha256 = _worktree_digest(path)
        worktree_status = ("hashed" if worktree_sha256 is not None else
                           "deleted" if "D" in status and not path.exists() else "unavailable")
        changed_files.append({
            "path": name,
            "status": status,
            "index_blob": index_blob,
            "worktree_sha256": worktree_sha256,
            "worktree_status": worktree_status,
        })

    changed_files.sort(key=lambda item: item["path"])
    digest_input = "\n".join(
        "\0".join((item["path"], item["status"], item["index_blob"] or "",
                    item["worktree_sha256"] or "<missing>"))
        for item in changed_files
    ).encode("utf-8", "surrogateescape")
    status = "no_commit" if commit is None else "dirty" if changed_files else "clean"
    complete = all(item["worktree_status"] != "unavailable" and "U" not in item["status"]
                   for item in changed_files)
    return {
        "schema": _SCHEMA,
        "status": status if complete else "incomplete",
        "commit": commit,
        "dirty": bool(changed_files),
        "changed_files": changed_files,
        "content_sha256": hashlib.sha256(digest_input).hexdigest() if complete else None,
    }
