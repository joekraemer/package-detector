"""Stdlib-only tests: config, zones, state machine, HA payload.
No camera, no model, no network. Run: python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    unittest.main()
