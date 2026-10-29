"""Stdlib-only tests: config, zones, state machine, HA payload.
No camera, no model, no network. Run: python -m unittest discover -s tests
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import packagedetector as pd  # noqa: E402
from packagedetector import (  # noqa: E402
    ConfigError, ha_payload, load_config, parse_cameras, parse_zones,
    update_state, zone_contains)


class TestCameras(unittest.TestCase):
    def test_parse(self):
        cams = parse_cameras("porch=rtsp://admin:p@10.0.0.2:554/h264Preview_01_sub,"
                             "garage=rtsp://admin:p@10.0.0.3:554/Preview_01_sub")
        self.assertEqual([c.name for c in cams], ["porch", "garage"])

    def test_rejects_non_rtsp(self):
        with self.assertRaises(ConfigError):
            parse_cameras("porch=http://10.0.0.2/snap.jpg")

    def test_rejects_duplicate(self):
        with self.assertRaises(ConfigError):
            parse_cameras("porch=rtsp://a,porch=rtsp://b")


class TestZones(unittest.TestCase):
    def test_parse(self):
        zones = parse_zones("porch=0,0.5,1,1,garage=0,0.3,1,1")
        self.assertEqual(zones["porch"], (0.0, 0.5, 1.0, 1.0))
        self.assertEqual(zones["garage"], (0.0, 0.3, 1.0, 1.0))

    def test_centre_inside(self):
        zone = (0.0, 0.5, 1.0, 1.0)
        self.assertTrue(zone_contains(zone, (0.4, 0.6, 0.6, 0.9)))
        self.assertFalse(zone_contains(zone, (0.4, 0.1, 0.6, 0.3)))

    def test_no_zone_is_full_frame(self):
        self.assertTrue(zone_contains(None, (0.0, 0.0, 1.0, 1.0)))

    def test_unknown_camera_rejected(self):
        with self.assertRaises(ConfigError):
            load_config({"CAMERAS": "porch=rtsp://a", "ZONES": "nope=0,0,1,1"})


class TestState(unittest.TestCase):
    def test_needs_two_hits_to_flip_on(self):
        present, hits, misses, changed = update_state(False, 0, 0, True, 2, 2)
        self.assertEqual((present, changed), (False, False))
        present, hits, misses, changed = update_state(present, hits, misses, True, 2, 2)
        self.assertEqual((present, changed), (True, True))

    def test_one_miss_does_not_clear(self):
        present, hits, misses, changed = update_state(True, 2, 0, False, 2, 2)
        self.assertEqual((present, changed), (True, False))
        present, hits, misses, changed = update_state(present, hits, misses, False, 2, 2)
        self.assertEqual((present, changed), (False, True))

    def test_detection_resets_misses(self):
        _, hits, misses, _ = update_state(True, 2, 1, True, 2, 2)
        self.assertEqual(misses, 0)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config({"CAMERAS": "porch=rtsp://a"})
        self.assertEqual(cfg.poll_interval_seconds, 15)
        self.assertEqual(cfg.confidence, 0.5)
        self.assertEqual(cfg.confirm_hits, 2)
        self.assertEqual(cfg.clear_misses, 5)

    def test_bad_poll_rejected(self):
        with self.assertRaises(ConfigError):
            load_config({"CAMERAS": "porch=rtsp://a", "POLL_INTERVAL_SECONDS": "10"})

    def test_live_needs_ha(self):
        with self.assertRaises(ConfigError):
            load_config({"CAMERAS": "porch=rtsp://a"}, require_ha=True)


class TestHA(unittest.TestCase):
    def test_payload(self):
        p = ha_payload("porch", True, 0.83, "2026-10-28T23:00:00+00:00")
        self.assertEqual(p["state"], "on")
        self.assertEqual(p["attributes"]["device_class"], "occupancy")


class TestLetterbox(unittest.TestCase):
    # A 640x360 (16:9) sub-stream frame letterboxes to 640x360 at y=140..500.
    def test_frame_edges_map_to_0_and_1(self):
        x1, y1, x2, y2 = pd.model_box_to_frame(320, 320, 640, 360, 640, 360)
        for got, want in ((x1, 0.0), (y1, 0.0), (x2, 1.0), (y2, 1.0)):
            self.assertAlmostEqual(got, want, places=6)

    def test_box_in_top_left_quarter(self):
        # canvas box centred at (160, 230), 320x180 = the frame's top-left quarter
        box = pd.model_box_to_frame(160, 230, 320, 180, 640, 360)
        for got, want in zip(box, (0.0, 0.0, 0.5, 0.5)):
            self.assertAlmostEqual(got, want, places=6)

    def test_portrait_pads_x_and_clips(self):
        scale, px, py = pd.letterbox_params(360, 640)
        self.assertEqual((scale, px, py), (1.0, 140, 0))
        box = pd.model_box_to_frame(100, 320, 100, 100, 360, 640)  # left of the image
        self.assertEqual(box[0], 0.0)


class TestRedaction(unittest.TestCase):
    def test_ffmpeg_error_line(self):
        line = "rtsp://admin:S3cret!@192.168.1.20:554/h264Preview_01_sub: Connection refused"
        out = pd.redact_url(line)
        self.assertNotIn("S3cret", out)
        self.assertIn("rtsp://***@192.168.1.20:554", out)

    def test_config_errors_hide_password(self):
        with self.assertRaises(ConfigError) as cm:
            parse_cameras("porch=http://admin:S3cret@10.0.0.2/x")
        self.assertNotIn("S3cret", str(cm.exception))
        with self.assertRaises(ConfigError) as cm:
            parse_cameras("rtsp://admin:S3cret@10.0.0.2/x")
        self.assertNotIn("S3cret", str(cm.exception))

    def test_failed_capture_removes_temp_file_and_redacts(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "ffmpeg")
            with open(fake, "w") as f:
                # print the input URL like ffmpeg does, fail, leave the output file
                f.write('#!/bin/sh\nfor a; do last=$a; done\n: > "$last"\n'
                        'echo "$7: Connection refused" >&2\nexit 1\n')
            os.chmod(fake, 0o755)
            before = set(os.listdir(tempfile.gettempdir()))
            with mock.patch.dict(os.environ, {"PATH": d + os.pathsep + os.environ["PATH"]}):
                snap, err = pd.capture_snapshot("rtsp://admin:S3cret@10.0.0.2/x", 5)
            after = set(os.listdir(tempfile.gettempdir()))
        self.assertIsNone(snap)
        self.assertNotIn("S3cret", err)
        self.assertIn("Connection refused", err)
        self.assertFalse([n for n in after - before if n.endswith(".jpg")])


class TestCameraNames(unittest.TestCase):
    def test_entity_safe_names_only(self):
        for bad in ("Front", "front-door", "front door"):
            with self.assertRaises(ConfigError):
                parse_cameras(f"{bad}=rtsp://a")
        self.assertEqual(parse_cameras("front_door2=rtsp://a")[0].name, "front_door2")


class TestHAPostDecision(unittest.TestCase):
    def test_rules(self):
        self.assertTrue(pd.needs_ha_post(True, True, 1000.0, 1001.0, 300))    # change
        self.assertTrue(pd.needs_ha_post(False, False, 1000.0, 1001.0, 300))  # last failed
        self.assertTrue(pd.needs_ha_post(False, True, None, 1001.0, 300))     # never posted
        self.assertTrue(pd.needs_ha_post(False, True, 1000.0, 1300.0, 300))   # refresh due
        self.assertFalse(pd.needs_ha_post(False, True, 1000.0, 1299.0, 300))  # quiet


class TestPollOnce(unittest.TestCase):
    """poll_once end to end with capture, inference and HA stubbed."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        model = os.path.join(self.dir, "m.onnx")
        open(model, "w").close()
        self.cfg = load_config({"CAMERAS": "porch=rtsp://a", "DATA_DIR": self.dir,
                                "MODEL_PATH": model, "CONFIRM_HITS": "1",
                                "CLEAR_MISSES": "1", "HA_URL": "http://ha", "HA_TOKEN": "t"})
        self.snaps = []
        self.seen = [True]
        self.posts = []
        self.ha_err = [""]

        def capture(url, timeout):
            fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.dir)
            os.write(fd, b"jpg")
            os.close(fd)
            self.snaps.append(path)
            return path, ""

        def detect(path, model, conf):
            if self.seen[0]:
                return [{"box": (0.4, 0.4, 0.6, 0.6), "confidence": 0.9}]
            return []

        def post(cfg, cam, present, conf, last_seen):
            self.posts.append((present, last_seen))
            return self.ha_err[0]

        for name, fn in (("capture_snapshot", capture), ("detect", detect),
                         ("post_to_ha", post)):
            p = mock.patch.object(pd, name, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(pd, "annotate")
        p.start()
        self.addCleanup(p.stop)

    def state(self):
        conn = pd.init_db(self.dir)
        try:
            return pd.load_state(conn, "porch")
        finally:
            conn.close()

    def test_temp_snapshots_removed_even_when_inference_fails(self):
        with mock.patch.object(pd, "detect", side_effect=RuntimeError("bad model")):
            self.assertEqual(pd.poll_once(self.cfg), 1)
        pd.poll_once(self.cfg, dry_run=True)
        pd.poll_once(self.cfg)
        self.assertEqual(len(self.snaps), 3)
        self.assertFalse(any(os.path.exists(s) for s in self.snaps))

    def test_since_and_last_seen_survive_quiet_polls(self):
        pd.poll_once(self.cfg)                       # arrives
        s1 = self.state()
        self.assertTrue(s1.present)
        pd.poll_once(self.cfg)                       # still there
        s2 = self.state()
        self.assertEqual(s2.since, s1.since)
        self.assertIsNotNone(s2.last_seen)
        self.seen[0] = False
        pd.poll_once(self.cfg)                       # leaves
        s3 = self.state()
        self.assertFalse(s3.present)
        self.assertEqual(s3.last_seen, s2.last_seen)
        self.assertEqual(self.posts[-1], (False, s2.last_seen))

    def test_failed_post_is_retried_next_poll(self):
        self.ha_err[0] = "HA POST failed: 503"
        self.assertEqual(pd.poll_once(self.cfg), 1)  # change, POST fails
        self.ha_err[0] = ""
        self.assertEqual(pd.poll_once(self.cfg), 0)  # no change, but retried
        self.assertEqual(len(self.posts), 2)
        self.assertTrue(self.state().ha_synced)
        pd.poll_once(self.cfg)                       # synced and quiet: no POST
        self.assertEqual(len(self.posts), 2)

    def test_refresh_reposts_unchanged_state(self):
        pd.poll_once(self.cfg)
        with mock.patch.object(pd.time, "time", return_value=time.time() + 301):
            pd.poll_once(self.cfg)
        self.assertEqual(len(self.posts), 2)

    def test_old_database_is_migrated(self):
        import sqlite3
        conn = sqlite3.connect(os.path.join(self.dir, "package-detector.sqlite"))
        conn.execute("""CREATE TABLE camera_state (camera TEXT PRIMARY KEY, present INTEGER,
                        hits INTEGER, misses INTEGER, last_seen TEXT, since TEXT)""")
        conn.execute("INSERT INTO camera_state VALUES ('porch', 1, 3, 0, 'x', 'y')")
        conn.commit()
        conn.close()
        s = self.state()
        self.assertTrue(s.present)
        self.assertFalse(s.ha_synced)   # unknown, so the next poll re-posts


class TestLoopInterval(unittest.TestCase):
    def test_interval_source(self):
        import loop
        self.assertEqual(loop._interval({}), 15)
        self.assertEqual(loop._interval({"POLL_INTERVAL_SECONDS": "30"}), 30)
        self.assertEqual(loop._interval({"POLL_INTERVAL_SECONDS": "30", "LOOP_INTERVAL": "60"}), 60)


if __name__ == "__main__":
    unittest.main()
