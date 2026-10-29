#!/usr/bin/env python3
"""Run the app on a fixed interval, forever. This is the container's PID 1.

Copied in shape from birdlisten's loop.py, with one difference that
matters: birdlisten's main() runs forever in stream mode, so its loop
interval is 1 s (restart quickly after a fatal error). Here main() is
ONE poll, so the loop interval *is* the poll interval. It comes from
POLL_INTERVAL_SECONDS in .env (default 15); LOOP_INTERVAL, if set,
overrides it. Note the loop sleeps *after* a poll finishes, so the real
period is poll duration + interval, not a strict 15 s cadence.

Why a loop and not cron: the container is then an ordinary long-running
service. Docker's `restart: unless-stopped` restarts it if it dies,
`docker compose logs` has everything, and there is no host-side schedule
to maintain per app. One process, one job, one log stream.

Environment:
  LOOP_INTERVAL   seconds between polls (default: POLL_INTERVAL_SECONDS, else 15)
  LOOP_JITTER     random extra 0..N seconds per run (default 0)
  LOOP_ONCE       set to 1 to run a single poll and exit (handy for testing)
  RUN_ARGS        extra CLI args passed to the app, space separated

A failing run is logged with its traceback and the loop continues. Only a
failure to even import the app is fatal (that is a packaging bug, and a
crash-loop is the right signal for it).
"""

from __future__ import annotations

import logging
import os
import random
import shlex
import signal
import sys
import time
import traceback

# ---- change this one line per app ------------------------------------------
from packagedetector import main as app_main  # noqa: E402
# ----------------------------------------------------------------------------

log = logging.getLogger("loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_stop = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True


def _interval(env) -> int:
    """Seconds between polls: LOOP_INTERVAL if set (fleet convention),
    else the app's own POLL_INTERVAL_SECONDS, else 15. One number to set."""
    raw = env.get("LOOP_INTERVAL") or env.get("POLL_INTERVAL_SECONDS") or "15"
    return max(int(raw), 1)


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    interval = _interval(os.environ)
    jitter = int(os.environ.get("LOOP_JITTER", "0"))
    once = os.environ.get("LOOP_ONCE", "") == "1"
    run_args = shlex.split(os.environ.get("RUN_ARGS", ""))
    while True:
        started = time.monotonic()
        sys.argv = [sys.argv[0], *run_args]
        try:
            rc = app_main()
            log.info("run finished rc=%s after %.1fs", rc, time.monotonic() - started)
        except SystemExit as exc:  # argparse / sys.exit inside the app
            log.info("run finished rc=%s after %.1fs", exc.code, time.monotonic() - started)
        except Exception:  # noqa: BLE001 -- log and keep the loop alive
            log.error("run failed after %.1fs:\n%s", time.monotonic() - started, traceback.format_exc())
        if once or _stop:
            return 0
        delay = interval + random.randint(0, max(jitter, 0))
        log.info("next run in %ss", delay)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and not _stop:
            time.sleep(1)
        if _stop:
            return 0


if __name__ == "__main__":
    sys.exit(main())
