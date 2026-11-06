"""replay.py: alert rules over a poll log, verdicts, deliveries, uncertain stretches.
Needs numpy/Pillow (it imports train); skipped where they are missing."""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import replay
    import train
except ImportError as exc:  # stdlib-only environments (the runtime image has both)
    replay = None
    SKIP = str(exc)

T0 = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)


def polls(*ps, step=20, start=T0):
    return [(start + timedelta(seconds=i * step), p) for i, p in enumerate(ps)]


@unittest.skipIf(replay is None, "needs numpy/Pillow")
class TestReplay(unittest.TestCase):
    def test_hits_misses_and_cutoff(self):
        seq = polls(0.1, 0.75, 0.1, 0.75, 0.75, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1)
        r = replay.replay(seq, 0.7, hits=2, misses=5, cooldown_s=0)
        self.assertEqual([a.at for a in r.alerts], [seq[4][0]])  # one stray hit is not enough
        self.assertEqual(r.on_polls, [seq[i][0] for i in range(4, 10)])  # off after 5 misses
        self.assertEqual(replay.replay(seq, 0.8, 2, 5, 0).alerts, [])

    def test_cooldown_counts_from_last_notification(self):
        on, off = [0.9, 0.9], [0.1] * 5
        seq = polls(*(on + off + on + off), step=60)  # second arrival 7 min after the first
        self.assertEqual(len(replay.replay(seq, 0.5, 2, 5, 1800).alerts), 1)
        r = replay.replay(seq, 0.5, 2, 5, 300)
        self.assertEqual(len(r.alerts), 2)
        self.assertEqual(len(r.switches_on), 2)

    def test_judge(self):
        truth = [(T0 - timedelta(minutes=20), "empty", "e.jpg"),
                 (T0 + timedelta(minutes=25), "package", "p.jpg")]
        alerts = [replay.Alert(T0, 0.9),                          # package 25 min later: true
                  replay.Alert(T0 - timedelta(minutes=40), 0.8),  # empty 20 min later: false
                  replay.Alert(T0 + timedelta(hours=5), 0.8)]     # nothing near
        replay.judge(alerts, truth)
        self.assertEqual([(a.verdict, a.evidence) for a in alerts],
                         [("true", "p.jpg"), ("false", "e.jpg"), ("unverified", "")])

    def test_deliveries_and_caught(self):
        t = [T0, T0 + timedelta(minutes=20), T0 + timedelta(minutes=40), T0 + timedelta(hours=3)]
        truth = [(x, "package", f"{i}.jpg") for i, x in enumerate(t)] + [(T0 + timedelta(hours=1), "empty", "e")]
        dels = replay.deliveries(truth, T0 - timedelta(days=1), T0 + timedelta(days=1))
        self.assertEqual([(a, b, n) for a, b, n in dels],
                         [(t[0], t[2], 3), (t[3], t[3], 1)])
        self.assertEqual(replay.deliveries(truth, T0 + timedelta(hours=2), T0 + timedelta(days=1)),
                         [(t[3], t[3], 1)])  # outside the polled period: not counted
        r = replay.Replay(0.7, on_polls=[T0 + timedelta(minutes=41)])
        self.assertTrue(replay.caught(r, t[0], t[2]))
        self.assertFalse(replay.caught(r, t[3], t[3]))

    def test_uncertain_stretches(self):
        seq = polls(0.5, 0.6, 0.7, 0.9, 0.5, 0.5, 0.1, 0.45, 0.45, 0.45)
        seq += polls(0.5, 0.5, start=seq[-1][0] + timedelta(minutes=10))  # a gap breaks the run
        got = replay.uncertain_stretches(seq, 0.4, 0.8, min_polls=3)
        self.assertEqual([(a, b, n, p) for a, b, n, p in got],
                         [(seq[0][0], seq[2][0], 3, 0.7), (seq[7][0], seq[9][0], 3, 0.45)])
        tail = seq + polls(0.6, start=seq[-1][0] + timedelta(seconds=20))  # the log ends mid-stretch
        self.assertEqual(replay.uncertain_stretches(tail, 0.4, 0.8, 3)[-1][2:], (3, 0.6))

    def test_end_to_end_on_a_tiny_db(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        db = d / "pd.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE polls (ts TEXT, camera TEXT, detected INTEGER, confidence REAL, present INTEGER)")
        for at, p in polls(0.1, 0.95, 0.95, 0.95, *([0.1] * 36), 0.85, 0.85, step=60):
            conn.execute("INSERT INTO polls VALUES (?, 'doorbell', 0, ?, 0)", (at.isoformat(), p))
        conn.commit()
        conn.close()
        frames = d / "frames"
        day = (frames / "2026-11-05" / "doorbell")
        day.mkdir(parents=True)
        rows = []
        for minute, reason in ((2, "after_30s"), (45, "baseline")):
            at = (T0 + timedelta(minutes=minute)).astimezone()
            f = f"2026-11-05/doorbell/{at.strftime('%H%M%S')}_{reason}.jpg"
            (frames / f).write_bytes(b"x")
            rows.append({"file": f, "camera": "doorbell", "ts": at.isoformat(), "reason": reason, "phase": "day"})
        (frames / "2026-11-05" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (frames / "labels.jsonl").write_text(json.dumps({"file": rows[0]["file"], "label": "package"}) + "\n")
        args = replay.argparse.Namespace(db=db, data=frames, camera="doorbell", cutoff=[0.8, 0.9], hits=2,
                                         misses=5, cooldown=0, band=(0.4, 0.8), min_polls=3, since=None)
        rep = replay.run(args)
        self.assertEqual(rep["cutoffs"][0.8]["true"], 1)    # the 0.95 run, package frame at +2 min
        self.assertEqual(rep["cutoffs"][0.8]["false"], 1)   # the 0.85 pair at +41 min; empty baseline at +45
        self.assertEqual(rep["cutoffs"][0.9]["alerts"], 1)
        self.assertEqual(rep["cutoffs"][0.9]["deliveries_caught"], 1)


@unittest.skipIf(replay is None, "needs numpy/Pillow")
class TestTestDays(unittest.TestCase):
    def test_hold_out_and_read(self):
        labels = {"2026-11-04/doorbell/1.jpg": "package", "2026-11-05/doorbell/2.jpg": "empty",
                  "2026-11-05/doorbell/3.jpg": "package"}
        kept, held = train.hold_out(labels, {"2026-11-05"})
        self.assertEqual(kept, {"2026-11-04/doorbell/1.jpg": "package"})
        self.assertEqual(held, ["2026-11-05/doorbell/2.jpg", "2026-11-05/doorbell/3.jpg"])
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        self.assertEqual(train.read_test_days(d), set())
        (d / train.TEST_DAYS_FILE).write_text('{"doorbell": ["2026-11-05"], "garage": []}')
        self.assertEqual(train.read_test_days(d), {("doorbell", "2026-11-05")})


if __name__ == "__main__":
    unittest.main()
