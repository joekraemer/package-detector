"""Tests for review.py: reference selection, change score, labels, server.
No real frames: tiny synthetic JPEGs. Run: python -m unittest discover -s tests
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import review  # noqa: E402

TZ = timezone(timedelta(hours=-8))


def row(file, ts, reason="baseline", event=None, camera="doorbell"):
    dt = datetime.fromisoformat(ts).replace(tzinfo=TZ)
    r = {"file": file, "camera": camera, "ts": dt.isoformat(), "reason": reason,
         "phase": "day", "dt": dt}
    if event:
        r["event"] = event
    return r


class PreviousBaseline(unittest.TestCase):
    def setUp(self):
        self.bases = [row("a", "2026-11-03T10:00:00"), row("b", "2026-11-03T10:20:00"),
                      row("c", "2026-11-03T10:40:00")]

    def test_visit_frame_uses_baseline_before_visit_start(self):
        # after_120s at 10:23 belongs to a visit that began at 10:19, so the
        # 10:20 baseline may already show the visitor: use 10:00.
        r = row("x", "2026-11-03T10:23:00", "after_120s", event="20261103T101900")
        self.assertEqual(review.previous_baseline(r, self.bases)["file"], "a")

    def test_baseline_uses_the_one_before_it(self):
        self.assertEqual(review.previous_baseline(self.bases[2], self.bases)["file"], "b")

    def test_too_old_reference_is_dropped(self):
        r = row("x", "2026-11-03T13:00:00", "arrive", event="20261103T130000")
        self.assertIsNone(review.previous_baseline(r, self.bases))


class ChangeScore(unittest.TestCase):
    def test_brightness_change_alone_scores_low(self):
        ref = np.full((120, 160), 100, np.float32)
        ref[60:, :80] = 60  # some structure
        self.assertLess(review.change_score(ref * 1.4, ref, "doorbell"), 0.05)

    def test_new_object_scores_high(self):
        ref = np.full((120, 160), 100, np.float32)
        img = ref.copy()
        img[80:120, 40:100] = 200  # a bright box on the floor
        self.assertGreater(review.change_score(img, ref, "doorbell"), 0.1)

    def test_masked_top_strip_is_ignored(self):
        ref = np.full((120, 160), 100, np.float32)
        img = ref.copy()
        img[:30] = 250  # garage street strip / timestamp
        self.assertEqual(review.change_score(img, ref, "garage"), 0.0)


class Zones(unittest.TestCase):
    def test_change_outside_zone_is_ignored(self):
        ref = np.full((120, 160), 100, np.float32)
        img = ref.copy()
        img[0:40, 0:40] = 220  # top-left: outside the floor zone
        zone = [0.25, 0.6, 0.75, 1.0]
        self.assertEqual(review.change_score(img, ref, "doorbell", zone), 0.0)
        img[90:110, 60:100] = 220  # inside it
        self.assertGreater(review.change_score(img, ref, "doorbell", zone), 0.1)

    def test_valid_zone(self):
        self.assertTrue(review.valid_zone([0.1, 0.2, 0.5, 0.9]))
        self.assertFalse(review.valid_zone([0.5, 0.2, 0.1, 0.9]))  # inverted
        self.assertFalse(review.valid_zone([0.1, 0.2, 0.11, 0.9]))  # too thin
        self.assertFalse(review.valid_zone("0,0,1,1"))


class ReferenceAcrossDays(unittest.TestCase):
    def test_same_day_baselines_never_used(self):
        cache = {"img": lambda f: np.zeros((2, 2), np.float32)}
        bases = [row("d1", "2026-11-03T10:00:00"), row("d2", "2026-11-03T10:20:00"),
                 row("d3", "2026-11-03T10:40:00")]
        self.assertIsNone(review.reference_for(row("x", "2026-11-03T10:30:00"), bases, cache))


class Labels(unittest.TestCase):
    def test_last_line_wins_and_undo_removes(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        p = d / "labels.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in [
            {"file": "a", "label": "empty"}, {"file": "a", "label": "package"},
            {"file": "b", "label": "unsure"}, {"file": "b", "label": None}]) + "\n")
        self.assertEqual(review.read_labels(p), {"a": "package"})


class EndToEnd(unittest.TestCase):
    """Score two synthetic days, then hit the server."""

    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.data)
        rows = []
        for day in ("2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04"):
            (self.data / day / "doorbell").mkdir(parents=True)
            for hhmm in ("1000", "1020", "1040"):
                f = f"{day}/doorbell/{hhmm}00_baseline.jpg"
                arr = np.full((480, 640, 3), 90, np.uint8)
                if day == "2026-11-04" and hhmm == "1040":
                    arr[360:470, 200:400] = 230  # a package appears
                Image.fromarray(arr).save(self.data / f)
                rows.append({"file": f, "camera": "doorbell",
                             "ts": f"{day}T{hhmm[:2]}:{hhmm[2:]}:00-08:00",
                             "reason": "baseline", "phase": "day"})
        for day in {r["file"][:10] for r in rows}:
            (self.data / day / "index.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows if r["file"].startswith(day)))

    def test_zone_save_rescores_and_rejects_bad_input(self):
        zones_path = self.data / "zones.json"
        srv = ThreadingHTTPServer(("127.0.0.1", 0), review.make_handler(self.data, 0, zones_path))
        port = srv.server_address[1]
        srv.RequestHandlerClass = review.make_handler(self.data, port, zones_path)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{port}"

        def post(body):
            return urllib.request.urlopen(urllib.request.Request(
                base + "/api/zones", method="POST", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}))

        # Zone away from the "package": that frame should no longer stand out.
        post({"doorbell": [0.0, 0.1, 0.25, 0.5]})
        self.assertEqual(review.load_zones(zones_path), {"doorbell": [0.0, 0.1, 0.25, 0.5]})
        scores = {json.loads(l)["file"]: json.loads(l)["score"]
                  for l in (self.data / "scores.jsonl").read_text().splitlines()}
        self.assertLess(scores["2026-11-04/doorbell/104000_baseline.jpg"], 0.01)
        for bad in ({"doorbell": [0.5, 0.5, 0.4, 0.9]}, {"doorbell": [0, 0, 2, 1]}, {"../x": [0, 0, 1, 1]}, []):
            with self.assertRaises(urllib.error.HTTPError) as e:
                post(bad)
            self.assertEqual(e.exception.code, 400)
        page = urllib.request.urlopen(base + "/api/zones")
        self.assertIn("doorbell", json.load(page)["samples"])

    def test_package_frame_ranks_first_and_labels_persist(self):
        self.assertEqual(review.cmd_score(self.data, self.data / "no-zones.json"), 0)
        scores = [json.loads(l) for l in (self.data / "scores.jsonl").read_text().splitlines()]
        top = max(scores, key=lambda r: r["score"] or 0)
        self.assertEqual(top["file"], "2026-11-04/doorbell/104000_baseline.jpg")

        nz = self.data / "no-zones.json"
        srv = ThreadingHTTPServer(("127.0.0.1", 0), review.make_handler(self.data, 0, nz))
        port = srv.server_address[1]
        srv.RequestHandlerClass = review.make_handler(self.data, port, nz)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{port}"

        frames = json.load(urllib.request.urlopen(base + "/api/frames"))
        self.assertEqual(frames[0]["file"], top["file"])

        req = urllib.request.Request(base + "/api/label", method="POST",
                                     data=json.dumps({"file": top["file"], "label": "package"}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)
        self.assertEqual(review.read_labels(self.data / "labels.jsonl"), {top["file"]: "package"})

        # A saved label schedules the data backup; a rejected one does not.
        backups = []
        srv.RequestHandlerClass = review.make_handler(self.data, port, nz, on_label=lambda: backups.append(1))
        urllib.request.urlopen(urllib.request.Request(
            base + "/api/label", method="POST", headers={"Content-Type": "application/json"},
            data=json.dumps({"file": top["file"], "label": "empty"}).encode()))
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(
                base + "/api/label", method="POST", headers={"Content-Type": "application/json"},
                data=json.dumps({"file": top["file"], "label": "maybe"}).encode()))
        self.assertEqual(backups, [1])

        # Path traversal and foreign Host headers are refused.
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/img/..%2f..%2fetc%2fpasswd")
        self.assertEqual(e.exception.code, 404)
        evil = urllib.request.Request(base + "/api/frames", headers={"Host": f"evil.example:{port}"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(evil)
        self.assertEqual(e.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
