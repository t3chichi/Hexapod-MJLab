"""Git provenance snapshots cover actual code contents without touching index."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hexapod_mjlab.common.provenance import capture_code_provenance


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Provenance Test")
    _git(path, "config", "user.email", "provenance@example.invalid")
    (path / "tracked.py").write_text("value = 1\n")
    (path / "delete me.py").write_text("remove me\n")
    (path / ".gitignore").write_text("logs/\n*.pt\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "baseline")
    return path


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_clean_and_non_git_states_are_explicit(self):
        repo = _repository(self.root / "repo")
        clean = capture_code_provenance(repo / "tracked.py")
        self.assertEqual(clean["status"], "clean")
        self.assertIs(clean["dirty"], False)
        self.assertEqual(clean["changed_files"], [])
        self.assertEqual(clean["commit"], _git(repo, "rev-parse", "HEAD"))

        outside = self.root / "wheel" / "module.py"
        outside.parent.mkdir()
        outside.write_text("pass\n")
        missing = capture_code_provenance(outside)
        self.assertEqual(missing["status"], "not_git_checkout")
        self.assertIsNone(missing["dirty"])
        self.assertIsNone(missing["commit"])

        empty_repo = self.root / "empty-repo"
        empty_repo.mkdir()
        _git(empty_repo, "init", "-q")
        (empty_repo / "pending.py").write_text("value = 1\n")
        no_commit = capture_code_provenance(empty_repo / "pending.py")
        self.assertEqual(no_commit["status"], "no_commit")
        self.assertIs(no_commit["dirty"], True)
        self.assertIsNone(no_commit["commit"])

    def test_staged_unstaged_untracked_deletion_and_ignored_paths(self):
        repo = _repository(self.root / "repo")
        tracked = repo / "tracked.py"
        tracked.write_text("value = 2\n")
        _git(repo, "add", "tracked.py")
        index_before = (repo / ".git" / "index").read_bytes()
        tracked.write_text("value = 3\n")
        (repo / "new source with spaces.py").write_text("answer = 42\n")
        (repo / "delete me.py").unlink()
        (repo / "logs").mkdir()
        (repo / "logs" / "run.log").write_text("must not be inspected\n")
        (repo / "checkpoint.pt").write_bytes(b"must not be inspected")

        snapshot = capture_code_provenance(repo / "tracked.py")
        self.assertEqual(snapshot["status"], "dirty")
        self.assertIs(snapshot["dirty"], True)
        by_path = {item["path"]: item for item in snapshot["changed_files"]}
        self.assertEqual(set(by_path), {"tracked.py", "new source with spaces.py", "delete me.py"})
        self.assertEqual(by_path["tracked.py"]["status"], "MM")
        self.assertTrue(by_path["tracked.py"]["index_blob"])
        self.assertTrue(by_path["tracked.py"]["worktree_sha256"])
        self.assertEqual(by_path["new source with spaces.py"]["status"], "??")
        self.assertTrue(by_path["new source with spaces.py"]["worktree_sha256"])
        self.assertIsNone(by_path["delete me.py"]["worktree_sha256"])
        self.assertEqual((repo / ".git" / "index").read_bytes(), index_before)

        first_digest = snapshot["content_sha256"]
        (repo / "new source with spaces.py").write_text("answer = 43\n")
        changed = capture_code_provenance(repo / "tracked.py")
        self.assertNotEqual(changed["content_sha256"], first_digest)
        self.assertNotEqual(changed["changed_files"], snapshot["changed_files"])

    def test_staged_deletion_keeps_deleted_state(self):
        repo = _repository(self.root / "repo")
        _git(repo, "rm", "delete me.py", "-q")
        snapshot = capture_code_provenance(repo / "tracked.py")
        entry = next(item for item in snapshot["changed_files"] if item["path"] == "delete me.py")
        self.assertEqual(entry["status"], "D ")
        self.assertIsNone(entry["index_blob"])
        self.assertIsNone(entry["worktree_sha256"])

    def test_unavailable_git_and_unreadable_contents_do_not_claim_clean_provenance(self):
        with patch("hexapod_mjlab.common.provenance.shutil.which", return_value=None):
            missing = capture_code_provenance(__file__)
        self.assertEqual(missing["status"], "git_unavailable")
        self.assertIsNone(missing["dirty"])
        repo = _repository(self.root / "repo")
        (repo / "tracked.py").write_text("changed\n")
        with patch("hexapod_mjlab.common.provenance._worktree_digest", return_value=None):
            partial = capture_code_provenance(repo / "tracked.py")
        self.assertEqual(partial["status"], "incomplete")
        self.assertTrue(partial["dirty"])
        self.assertIsNone(partial["content_sha256"])


if __name__ == "__main__":
    unittest.main()
