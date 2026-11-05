"""Stdlib-only tests: config, zones, state machine, HA payload.
No camera, no model, no network. Run: python -m unittest discover -s tests
"""

import json
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


class TestClassifier(unittest.TestCase):
    """Zone-classifier path: sidecar meta, preprocessing, poll and HA_DRY_RUN."""

    META = {"classes": ["empty", "package"], "imgsz": 224,
            "cameras": {"porch": [0.1, 0.5, 0.9, 1.0]}}

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.model = os.path.join(self.dir, "package-cls.onnx")
        open(self.model, "w").close()
        with open(os.path.join(self.dir, "package-cls.json"), "w") as fh:
            json.dump(self.META, fh)
        self.p = [0.9]
        self.posts = []

        def capture(url, timeout):
            fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.dir)
            os.close(fd)
            return path, ""

        for name, fn in (("capture_snapshot", capture),
                         ("classify", lambda *a: self.p[0]),
                         ("post_to_ha", lambda *a: self.posts.append(a[2]) or "")):
            p = mock.patch.object(pd, name, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(pd, "annotate")
        p.start()
        self.addCleanup(p.stop)

    def cfg(self, **extra):
        env = {"CAMERAS": "porch=rtsp://a,garage=rtsp://b", "DATA_DIR": self.dir,
               "MODEL_PATH": self.model, "CONFIRM_HITS": "1", "CLEAR_MISSES": "1",
               "HA_URL": "http://ha", "HA_TOKEN": "t"}
        env.update(extra)
        return load_config(env, require_ha=True)

    def test_meta_is_read_and_validated(self):
        self.assertEqual(pd.load_model_meta(self.model)["imgsz"], 224)
        self.assertIsNone(pd.load_model_meta(os.path.join(self.dir, "other.onnx")))
        with open(os.path.join(self.dir, "package-cls.json"), "w") as fh:
            json.dump({"classes": ["a", "b"], "imgsz": 224, "cameras": {}}, fh)
        with self.assertRaises(ConfigError):
            pd.load_model_meta(self.model)

    def test_threshold_and_live_post(self):
        cfg = self.cfg()
        pd.poll_once(cfg)
        self.assertEqual(self.posts, [True])
        self.p[0] = 0.2
        pd.poll_once(cfg)
        self.assertEqual(self.posts, [True, False])

    def test_camera_without_zone_is_skipped(self):
        # garage has no zone in the meta: only porch is classified.
        with mock.patch.object(pd, "classify", return_value=0.9) as c:
            pd.poll_once(self.cfg())
        self.assertEqual(c.call_count, 1)
        self.assertEqual(c.call_args.args[2], (0.1, 0.5, 0.9, 1.0))

    def test_env_zone_overrides_meta(self):
        with mock.patch.object(pd, "classify", return_value=0.9) as c:
            pd.poll_once(self.cfg(ZONES="porch=0,0,0.5,0.5"))
        self.assertEqual(c.call_args.args[2], (0.0, 0.0, 0.5, 0.5))

    def test_ha_dry_run_never_posts_and_needs_no_token(self):
        cfg = load_config({"CAMERAS": "porch=rtsp://a", "DATA_DIR": self.dir,
                           "MODEL_PATH": self.model, "CONFIRM_HITS": "1",
                           "CLEAR_MISSES": "1", "HA_DRY_RUN": "1"}, require_ha=True)
        with self.assertLogs("packagedetector", "INFO") as logs:
            pd.poll_once(cfg)
            self.p[0] = 0.1
            pd.poll_once(cfg)
        self.assertEqual(self.posts, [])
        changes = [l for l in logs.output if "HA_DRY_RUN" in l]
        self.assertEqual(len(changes), 2)                       # on, then off
        self.assertFalse(any("error" in l.lower() for l in logs.output))  # fleet health
        conn = pd.init_db(self.dir)
        s = pd.load_state(conn, "porch")
        n = conn.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
        conn.close()
        self.assertEqual(n, 2)          # polls still recorded for review
        self.assertFalse(s.ha_synced)   # going live posts on the first poll

    def test_bad_dry_run_value(self):
        with self.assertRaises(ConfigError):
            self.cfg(HA_DRY_RUN="maybe")


class TestNotify(unittest.TestCase):
    """HA_NOTIFY: one push per arrival, retried on failure, cooldown, hello."""

    META = {"classes": ["empty", "package"], "imgsz": 224,
            "cameras": {"doorbell": [0.1, 0.5, 0.9, 1.0]}}

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.model = os.path.join(self.dir, "package-cls.onnx")
        open(self.model, "w").close()
        with open(os.path.join(self.dir, "package-cls.json"), "w") as fh:
            json.dump(self.META, fh)
        self.p = [0.9]
        self.pushes = []
        self.push_err = [""]
        self.clock = [1_000_000.0]

        def capture(url, timeout):
            fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.dir)
            os.close(fd)
            return path, ""

        def notify(cfg, payload):
            self.pushes.append(payload["title"])
            return self.push_err[0]

        for name, fn in (("capture_snapshot", capture),
                         ("classify", lambda *a: self.p[0]),
                         ("post_to_ha", lambda *a: ""),
                         ("notify_ha", notify)):
            p = mock.patch.object(pd, name, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)
        for name in ("annotate",):
            p = mock.patch.object(pd, name)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(pd.time, "time", side_effect=lambda: self.clock[0])
        p.start()
        self.addCleanup(p.stop)

    def cfg(self, **extra):
        env = {"CAMERAS": "doorbell=rtsp://a", "DATA_DIR": self.dir, "MODEL_PATH": self.model,
               "CONFIRM_HITS": "2", "CLEAR_MISSES": "2", "HA_URL": "http://ha", "HA_TOKEN": "t",
               "HA_NOTIFY": "notify.mobile_app_phone"}
        env.update(extra)
        return load_config(env, require_ha=True)

    def poll(self, cfg, p, n=1, dt=20):
        self.p[0] = p
        for _ in range(n):
            pd.poll_once(cfg)
            self.clock[0] += dt

    def arrivals(self):
        return [t for t in self.pushes if t.startswith("Package at")]

    def test_parse(self):
        self.assertEqual(pd.parse_notify("mobile_app_a, notify.b"),
                         ["notify.mobile_app_a", "notify.b"])
        self.assertEqual(pd.parse_notify(""), [])
        for bad in ("notify.a/../b", "light.turn_on.x", "notify.A"):
            with self.assertRaises(ConfigError):
                pd.parse_notify(bad)

    def test_one_push_per_arrival_after_confirmation(self):
        cfg = self.cfg()
        self.poll(cfg, 0.1, 2)
        self.assertEqual(self.pushes, ["Package alerts are on"])   # hello, once
        self.poll(cfg, 0.9)                 # 1st hit: not confirmed yet
        self.assertEqual(self.arrivals(), [])
        self.poll(cfg, 0.9, 5)              # confirmed, then it stays there
        self.assertEqual(self.arrivals(), ["Package at the doorbell"])
        self.assertEqual(self.pushes.count("Package alerts are on"), 1)

    def test_failed_push_is_retried_while_present_and_dropped_once_gone(self):
        cfg = self.cfg()
        self.push_err[0] = "notify.mobile_app_phone: 500"
        self.poll(cfg, 0.9, 2)              # hello fails twice; confirmed, arrival push fails
        self.push_err[0] = ""
        self.poll(cfg, 0.9)                 # both retried and sent
        self.assertEqual(self.arrivals(), ["Package at the doorbell"] * 2)  # 1 failed + 1 sent
        self.poll(cfg, 0.9, 3)
        self.assertEqual(len(self.arrivals()), 2)                    # no more
        conn = pd.init_db(self.dir)
        hello = conn.execute("SELECT value FROM meta WHERE key='notify_hello'").fetchone()
        conn.close()
        self.assertEqual(hello[0], "notify.mobile_app_phone")

    def test_pending_cleared_when_package_leaves_before_push_works(self):
        cfg = self.cfg()
        self.poll(cfg, 0.1)                 # hello sent
        self.push_err[0] = "down"
        self.poll(cfg, 0.9, 2)              # arrives, push fails
        self.poll(cfg, 0.1, 2)              # gone
        self.push_err[0] = ""
        before = len(self.arrivals())
        self.poll(cfg, 0.1, 3)
        self.assertEqual(len(self.arrivals()), before)               # nothing stale sent

    def test_cooldown(self):
        cfg = self.cfg(HA_NOTIFY_COOLDOWN_SECONDS="1800")
        self.poll(cfg, 0.9, 2)              # arrival 1
        self.poll(cfg, 0.1, 2)              # gone
        self.poll(cfg, 0.9, 2)              # arrival 2, minutes later: quiet
        self.assertEqual(len(self.arrivals()), 1)
        self.poll(cfg, 0.1, 2)
        self.clock[0] += 1800
        self.poll(cfg, 0.9, 2)              # arrival 3, after the cooldown
        self.assertEqual(len(self.arrivals()), 2)

    def test_dry_run_and_unset_never_push(self):
        with self.assertLogs("packagedetector", "INFO") as logs:
            self.poll(self.cfg(HA_DRY_RUN="1"), 0.9, 3)
        self.assertEqual(self.pushes, [])
        self.assertTrue(any("would notify" in l for l in logs.output))
        self.poll(self.cfg(HA_NOTIFY="", DATA_DIR=tempfile.mkdtemp(dir=self.dir)), 0.9, 3)
        self.assertEqual(self.pushes, [])

    def test_going_live_with_a_package_already_there_does_not_push(self):
        self.poll(self.cfg(HA_DRY_RUN="1"), 0.9, 3)     # present in shadow mode
        self.poll(self.cfg(), 0.9, 3)                    # flip to live
        self.assertEqual(self.arrivals(), [])
        self.assertEqual(self.pushes, ["Package alerts are on"])

    def test_restart_keeps_state(self):
        self.poll(self.cfg(), 0.9, 2)
        self.poll(self.cfg(), 0.9, 2)        # "new process": same db
        self.assertEqual(len(self.arrivals()), 1)
        self.assertEqual(self.pushes.count("Package alerts are on"), 1)

    def test_payload(self):
        from datetime import datetime as dt
        p = pd.notify_payload("doorbell", 0.973, dt(2026, 11, 5, 15, 4))
        self.assertEqual(p["title"], "Package at the doorbell")
        self.assertIn("3:04 PM", p["message"])
        self.assertIn("0.97", p["message"])
        self.assertEqual(p["data"]["tag"], "package-doorbell")


class TestClassifierInput(unittest.TestCase):
    """classifier_input must match train.py's crop + square + net resize."""

    def test_matches_training_pipeline(self):
        try:
            import numpy as np
            from PIL import Image
            import train
        except ImportError as exc:  # stdlib-only environments
            self.skipTest(f"needs numpy/Pillow: {exc}")
        rng = np.random.default_rng(0)
        im = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
        zone = (0.057, 0.581, 0.618, 0.995)
        got = pd.classifier_input(im, zone, 224)
        sq = train.to_square(train.zone_crop(im, list(zone)))
        want = np.asarray(sq.resize((224, 224), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
        self.assertEqual(got.shape, (1, 3, 224, 224))
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.array_equal(got, want))


class TestLoopInterval(unittest.TestCase):
    def test_interval_source(self):
        import loop
        self.assertEqual(loop._interval({}), 15)
        self.assertEqual(loop._interval({"POLL_INTERVAL_SECONDS": "30"}), 30)
        self.assertEqual(loop._interval({"POLL_INTERVAL_SECONDS": "30", "LOOP_INTERVAL": "60"}), 60)


if __name__ == "__main__":
    unittest.main()
