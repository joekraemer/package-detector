#!/usr/bin/env python3
"""datasync.py: back up labels to the private package-detector-data repo.

Labels exist nowhere else: the frames can be copied again from the fleet
Mac, a hand label cannot. So every labelling path ends here:

- review.py serve: 2 minutes after the last label (one commit per burst),
  and once more when the server stops;
- train.py prepare: before building a dataset;
- by hand: `python datasync.py "data: Relabel the Nov 5 delivery"`.

A backup runs the data repo's own sync.py (copies the label files and
the labelled frames), then commits and pushes if anything changed. It
never raises: a failed backup is reported and retried on the next run,
it must not lose the label that was just written or break training.

PACKAGE_DATA_REPO points at the data repo checkout (default: a sibling
`package-detector-data`); PACKAGE_DATA_SYNC=0 turns backups off.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
GIT_TIMEOUT_S = 120


def data_repo() -> Path:
    return Path(os.environ.get("PACKAGE_DATA_REPO") or REPO.parent / "package-detector-data")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)


def _unpushed(repo: Path) -> bool:
    """Local commits the remote does not have yet. origin/main moves on
    every successful push; missing (never pushed) counts as unpushed."""
    if _run(["git", "rev-parse", "--verify", "-q", "HEAD"], repo).returncode != 0:
        return False  # no commits at all
    r = _run(["git", "rev-list", "--count", "refs/remotes/origin/main..HEAD"], repo)
    return r.returncode != 0 or r.stdout.strip() != "0"


def backup(message: str | None = None, src: Path = REPO, repo: Path | None = None) -> str:
    """Sync, commit, push. Returns a one-line outcome; never raises."""
    if os.environ.get("PACKAGE_DATA_SYNC", "1").strip().lower() in ("0", "false", "no", "off"):
        return "data backup off (PACKAGE_DATA_SYNC=0)"
    repo = repo or data_repo()
    if not (repo / ".git").exists() or not (repo / "sync.py").is_file():
        return f"data backup skipped: no data repo at {repo}"
    message = message or f"data: Labels as of {datetime.now().astimezone():%Y-%m-%d %H:%M}"
    try:
        r = _run([sys.executable, "sync.py", "--src", str(src)], repo)
        if r.returncode != 0:
            return f"data backup failed in sync.py: {(r.stderr or r.stdout).strip().splitlines()[-1:]}"
        _run(["git", "add", "-A"], repo)
        if _run(["git", "diff", "--cached", "--quiet"], repo).returncode == 0:
            if not _unpushed(repo):
                return "data backup: nothing new"
            message = "earlier commit(s)"  # a past push failed: retry it now
        else:
            r = _run(["git", "commit", "-q", "-m", message], repo)
            if r.returncode != 0:
                return f"data backup failed to commit: {r.stderr.strip()[-200:]}"
        r = _run(["git", "push", "-q", "origin", "HEAD:refs/heads/main"], repo)
        if r.returncode == 0:
            _run(["git", "fetch", "-q", "origin", "main"], repo)  # keeps origin/main current
        if r.returncode != 0:
            # The commit stays local; the next backup pushes it.
            return f"data backup committed, push failed (next run retries): {r.stderr.strip()[-200:]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"data backup failed: {type(exc).__name__}"
    return f"data backup pushed: {message}"


class Debouncer:
    """Run `fn` once, `delay_s` after the LAST schedule() call.

    The review page writes one label per keypress; this turns a labelling
    session into one backup. Runs never overlap; a schedule() during a run
    queues one more run after it. flush() runs now (if anything is pending)
    and waits, for server shutdown."""

    def __init__(self, fn, delay_s: float = 120.0, log=print):
        self.fn, self.delay_s, self.log = fn, delay_s, log
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._pending = False

    def schedule(self) -> None:
        with self._lock:
            self._pending = True
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay_s, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._run_lock:
            with self._lock:
                if not self._pending:
                    return
                self._pending = False
            self.log(self.fn())

    def flush(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
        self._fire()


if __name__ == "__main__":
    print(backup(" ".join(sys.argv[1:]) or None))
