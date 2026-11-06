"""datasync.py: backup to the data repo (commit + push) and the debouncer.
Uses throwaway git repos with a local bare remote; no network."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasync  # noqa: E402

FAKE_SYNC = """import sys, pathlib
src = pathlib.Path(sys.argv[sys.argv.index("--src") + 1])
(pathlib.Path("labels.jsonl")).write_text((src / "labels.jsonl").read_text())
"""


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@unittest.skipIf(shutil.which("git") is None, "needs git")
class TestBackup(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.remote = self.tmp / "remote.git"
        git("init", "-q", "--bare", "-b", "main", str(self.remote), cwd=self.tmp)
        self.repo = self.tmp / "data"
        git("init", "-q", "-b", "main", str(self.repo), cwd=self.tmp)
        for k, v in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
            git("config", k, v, cwd=self.repo)
        git("remote", "add", "origin", str(self.remote), cwd=self.repo)
        (self.repo / "sync.py").write_text(FAKE_SYNC)
        self.src = self.tmp / "src"
        self.src.mkdir()
        (self.src / "labels.jsonl").write_text('{"file": "a.jpg", "label": "empty"}\n')
        env = mock.patch.dict(os.environ, {"PACKAGE_DATA_SYNC": "1"})
        env.start()
        self.addCleanup(env.stop)

    def remote_log(self):
        return git("--git-dir", str(self.remote), "log", "--format=%s", "main", cwd=self.tmp).split("\n")

    def test_commits_and_pushes_then_nothing_new(self):
        out = datasync.backup("data: First", src=self.src, repo=self.repo)
        self.assertEqual(out, "data backup pushed: data: First")
        self.assertEqual(self.remote_log()[0], "data: First")
        self.assertEqual(datasync.backup("data: Again", src=self.src, repo=self.repo), "data backup: nothing new")
        (self.src / "labels.jsonl").write_text('{"file": "a.jpg", "label": "package"}\n')
        datasync.backup("data: Relabel", src=self.src, repo=self.repo)
        self.assertEqual(self.remote_log()[:2], ["data: Relabel", "data: First"])

    def test_failed_push_keeps_the_commit_for_next_time(self):
        git("remote", "set-url", "origin", str(self.tmp / "missing.git"), cwd=self.repo)
        out = datasync.backup("data: Offline", src=self.src, repo=self.repo)
        self.assertIn("push failed", out)
        self.assertIn("data: Offline", git("log", "--format=%s", cwd=self.repo))
        git("remote", "set-url", "origin", str(self.remote), cwd=self.repo)
        # No new label, but the unpushed commit goes out on the next run.
        self.assertEqual(datasync.backup("data: Later", src=self.src, repo=self.repo),
                         "data backup pushed: earlier commit(s)")
        self.assertEqual(self.remote_log()[0], "data: Offline")
        self.assertEqual(datasync.backup(src=self.src, repo=self.repo), "data backup: nothing new")

    def test_off_missing_and_broken_sync_never_raise(self):
        with mock.patch.dict(os.environ, {"PACKAGE_DATA_SYNC": "0"}):
            self.assertIn("off", datasync.backup(src=self.src, repo=self.repo))
        self.assertIn("skipped", datasync.backup(src=self.src, repo=self.tmp / "nope"))
        (self.repo / "sync.py").write_text("raise SystemExit('boom')\n")
        self.assertIn("failed in sync.py", datasync.backup(src=self.src, repo=self.repo))
        self.assertEqual(git("--git-dir", str(self.remote), "branch", "--list", cwd=self.tmp), "")


class TestDebouncer(unittest.TestCase):
    def test_burst_becomes_one_run(self):
        calls = []
        d = datasync.Debouncer(lambda: calls.append(time.monotonic()) or "ok", delay_s=0.2, log=lambda m: None)
        for _ in range(5):
            d.schedule()
            time.sleep(0.05)
        time.sleep(0.5)
        self.assertEqual(len(calls), 1)
        d.schedule()
        time.sleep(0.4)
        self.assertEqual(len(calls), 2)

    def test_flush_runs_pending_now_and_only_then(self):
        calls = []
        d = datasync.Debouncer(lambda: calls.append(1) or "ok", delay_s=60, log=lambda m: None)
        d.flush()
        self.assertEqual(calls, [])  # nothing pending
        d.schedule()
        d.flush()
        self.assertEqual(calls, [1])
        time.sleep(0.05)
        self.assertEqual(calls, [1])  # the cancelled timer does not run it again

    def test_schedule_during_a_run_queues_one_more(self):
        started, release, calls = threading.Event(), threading.Event(), []

        def slow():
            calls.append(1)
            started.set()
            release.wait(2)
            return "ok"

        d = datasync.Debouncer(slow, delay_s=0.05, log=lambda m: None)
        d.schedule()
        self.assertTrue(started.wait(2))
        d.schedule()  # a label arrives while the backup runs
        release.set()
        time.sleep(0.4)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
