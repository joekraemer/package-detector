"""Stdlib-only tests for capture.py: config, sun phase, trigger tracker,
frame saving. No camera, no HA. Run: python -m unittest discover -s tests
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import capture  # noqa: E402
from capture import (  # noqa: E402
    TriggerTracker, load_capture_config, load_env_file, parse_triggers, sun_phase)
from packagedetector import ConfigError  # noqa: E402

CAMS = ("doorbell=rtsp://admin:p@10.0.0.2:554/h264Preview_01_sub,"
        "garage=rtsp://admin:p@10.0.0.3:554/Preview_01_sub")
NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def sun(state, rising=None, setting=None, changed=None):
    return {"state": state, "last_changed": changed.isoformat() if changed else None,
            "attributes": {"next_rising": rising.isoformat() if rising else None,
                           "next_setting": setting.isoformat() if setting else None}}


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_capture_config({"CAMERAS": CAMS})
        self.assertEqual([c.name for c in cfg.cameras], ["doorbell", "garage"])
        self.assertEqual(cfg.after_seconds, (30, 120))
        self.assertEqual(cfg.triggers, {})

    def test_triggers(self):
        t = parse_triggers("doorbell=binary_sensor.doorbell_person", ["doorbell", "garage"])
        self.assertEqual(t, {"doorbell": "binary_sensor.doorbell_person"})

    def test_trigger_unknown_camera(self):
        with self.assertRaises(ConfigError):
            parse_triggers("porch=binary_sensor.x", ["doorbell"])

    def test_trigger_must_be_binary_sensor(self):
        with self.assertRaises(ConfigError):
            parse_triggers("doorbell=sensor.x", ["doorbell"])

    def test_after_seconds_sorted_and_bounded(self):
        cfg = load_capture_config({"CAMERAS": CAMS, "CAPTURE_AFTER_SECONDS": "120, 30"})
        self.assertEqual(cfg.after_seconds, (30, 120))
        with self.assertRaises(ConfigError):
            load_capture_config({"CAMERAS": CAMS, "CAPTURE_AFTER_SECONDS": "0"})

    def test_env_file_does_not_override_and_keeps_equals(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = os.path.join(d, ".env")
        with open(path, "w") as fh:
            fh.write("# comment\nCAMERAS=a=rtsp://u:p@h/x\nHA_URL=\"http://ha:8123\"\nX=file\n")
        env = {"X": "real"}
        load_env_file(path, env)
        self.assertEqual(env["CAMERAS"], "a=rtsp://u:p@h/x")
        self.assertEqual(env["HA_URL"], "http://ha:8123")
        self.assertEqual(env["X"], "real")


class TestSunPhase(unittest.TestCase):
    def test_unknown_is_day(self):
        self.assertEqual(sun_phase(None, NOW), "day")
        self.assertEqual(sun_phase({"state": "unavailable"}, NOW), "day")

    def test_day_and_night(self):
        far = NOW + timedelta(hours=5)
        old = NOW - timedelta(hours=5)
        self.assertEqual(sun_phase(sun("above_horizon", far + timedelta(hours=9), far, old), NOW), "day")
        self.assertEqual(sun_phase(sun("below_horizon", far, far + timedelta(hours=9), old), NOW), "night")

    def test_twilight_before_sunset_and_after_sunrise(self):
        old = NOW - timedelta(hours=5)
        soon = NOW + timedelta(minutes=30)
        self.assertEqual(sun_phase(sun("above_horizon", NOW + timedelta(hours=14), soon, old), NOW),
                         "twilight")
        just_rose = NOW - timedelta(minutes=20)
        self.assertEqual(sun_phase(sun("above_horizon", NOW + timedelta(hours=23),
                                       NOW + timedelta(hours=10), just_rose), NOW), "twilight")


class TestTracker(unittest.TestCase):
    def test_first_state_is_not_an_arrival(self):
        t = TriggerTracker((30, 120), 12)
        self.assertEqual(t.update("doorbell", "on", 0), [])
        self.assertEqual(t.update("doorbell", "off", 10), [])  # no event id: nothing planned
        self.assertEqual(t.due(1000), [])

    def test_visit_produces_arrive_then_afters(self):
        t = TriggerTracker((30, 120), 12)
        t.update("doorbell", "off", 0)
        arrive = t.update("doorbell", "on", 100)
        self.assertEqual([s.reason for s in arrive], ["arrive"])
        t.update("doorbell", "off", 140)
        self.assertEqual(t.due(169), [])
        self.assertEqual([s.reason for s in t.due(170)], ["after_30s"])
        later = t.due(260)
        self.assertEqual([s.reason for s in later], ["after_120s"])
        self.assertEqual(later[0].event, arrive[0].event)

    def test_return_cancels_pending_afters(self):
        t = TriggerTracker((30, 120), 12)
        t.update("doorbell", "off", 0)
        t.update("doorbell", "on", 100)
        t.update("doorbell", "off", 110)
        t.update("doorbell", "on", 120)  # came back before 30 s
        self.assertEqual(t.due(1000), [])

    def test_unavailable_is_ignored(self):
        t = TriggerTracker((30,), 12)
        t.update("doorbell", "on", 0)
        t.update("doorbell", "unavailable", 1)
        self.assertEqual(t.update("doorbell", "on", 2), [])  # still the same visit
        t.update("doorbell", None, 3)
        self.assertEqual(t.update("doorbell", "on", 4), [])

    def test_hourly_cap(self):
        t = TriggerTracker((30,), 2)
        t.update("garage", "off", 0)
        arrivals = afters = 0
        for i in range(5):
            arrivals += len(t.update("garage", "on", 10 + i * 100))
            t.update("garage", "off", 50 + i * 100)
            afters += len(t.due(95 + i * 100))
        self.assertEqual(arrivals, 2)
        self.assertEqual(afters, 2)  # after shots only for the uncapped visits
        self.assertEqual(len(t.update("garage", "on", 10 + 4000)), 1)  # cap rolls off

    def test_cameras_are_independent(self):
        t = TriggerTracker((30,), 12)
        t.update("doorbell", "off", 0)
        t.update("garage", "off", 0)
        t.update("doorbell", "on", 10)
        t.update("doorbell", "off", 20)
        t.update("garage", "on", 25)  # must not cancel doorbell's after shot
        self.assertEqual([s.camera for s in t.due(100)], ["doorbell"])


class TestTake(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.cfg = load_capture_config({"CAMERAS": CAMS, "CAPTURE_DIR": self.dir,
                                        "CAPTURE_MIN_FREE_GB": "0"})

    def fake_snapshot(self, url, timeout):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.write(fd, b"\xff\xd8jpeg")
        os.close(fd)
        return path, ""

    def test_saves_and_indexes_without_credentials(self):
        when = datetime(2026, 10, 30, 14, 5, 9).astimezone()
        with mock.patch.object(capture.pd, "capture_snapshot", self.fake_snapshot):
            p1 = capture.take(self.cfg, self.cfg.cameras[0], "session", "session",
                              tag="brown-box", when=when)
            p2 = capture.take(self.cfg, self.cfg.cameras[0], "session", "session",
                              tag="brown-box", when=when)
        self.assertTrue(p1.endswith("2026-10-30/doorbell/140509_session.jpg"))
        self.assertTrue(p2.endswith("140509_session_2.jpg"))
        with open(os.path.join(self.dir, "2026-10-30", "index.jsonl")) as fh:
            text = fh.read()
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["tag"], "brown-box")
        self.assertEqual(rows[0]["camera"], "doorbell")
        self.assertNotIn("rtsp", text)

    def test_low_disk_skips(self):
        self.cfg.min_free_gb = 1e12
        with mock.patch.object(capture.pd, "capture_snapshot", self.fake_snapshot):
            self.assertIsNone(capture.take(self.cfg, self.cfg.cameras[0], "baseline", "day"))

    def test_failed_snapshot_writes_nothing(self):
        with mock.patch.object(capture.pd, "capture_snapshot", return_value=(None, "timeout")):
            self.assertIsNone(capture.take(self.cfg, self.cfg.cameras[0], "baseline", "day"))
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
