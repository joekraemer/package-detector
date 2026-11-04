"""Tests for train.py's labelling, split and paste helpers. No torch needed.
Run: python -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402


def r(file, reason="baseline", tag=None, camera="doorbell"):
    row = {"file": file, "camera": camera, "reason": reason, "ts": f"{file[:10]}T{file[20:22]}:{file[22:24]}:00-08:00"}
    if tag:
        row["tag"] = tag
    return row


ROWS = [
    r("2026-11-02/doorbell/100000_baseline.jpg"),
    r("2026-11-03/doorbell/100000_baseline.jpg"),
    r("2026-11-03/doorbell/180533_arrive.jpg", "arrive"),
    r("2026-11-04/doorbell/100000_baseline.jpg"),            # session day: no default label
    r("2026-11-04/doorbell/122520_session.jpg", "session", "brown-mailer"),
    r("2026-11-04/doorbell/124148_session.jpg", "session", "brown-mailer-flat-middle"),
    r("2026-11-04/doorbell/132907_session.jpg", "session", "small-box"),
    r("2026-11-04/doorbell/140000_session.jpg", "session", "empty"),
]


class Labels(unittest.TestCase):
    def test_rules(self):
        human = {"2026-11-03/doorbell/180533_arrive.jpg": "package",
                 "2026-11-04/doorbell/124148_session.jpg": "unsure"}
        got = train.resolve_labels(ROWS, human)
        self.assertEqual(got["2026-11-02/doorbell/100000_baseline.jpg"], "empty")
        self.assertEqual(got["2026-11-03/doorbell/180533_arrive.jpg"], "package")   # human wins
        self.assertNotIn("2026-11-04/doorbell/124148_session.jpg", got)              # unsure: left out
        self.assertNotIn("2026-11-04/doorbell/100000_baseline.jpg", got)             # session day
        self.assertEqual(got["2026-11-04/doorbell/122520_session.jpg"], "package")
        self.assertEqual(got["2026-11-04/doorbell/140000_session.jpg"], "empty")     # tag empty


class Split(unittest.TestCase):
    def test_val_day_keeps_its_real_delivery_and_newest_staged_go_to_val(self):
        labels = train.resolve_labels(ROWS, {"2026-11-03/doorbell/180533_arrive.jpg": "package"})
        sides = train.split(ROWS, labels)
        # Newest non-session day is 11-03: its negatives AND its delivery are val.
        self.assertEqual(sides["2026-11-03/doorbell/100000_baseline.jpg"], "val")
        self.assertEqual(sides["2026-11-03/doorbell/180533_arrive.jpg"], "val")
        self.assertEqual(sides["2026-11-02/doorbell/100000_baseline.jpg"], "train")
        # Of the staged positives, the newest share is val, the oldest train.
        self.assertEqual(sides["2026-11-04/doorbell/122520_session.jpg"], "train")
        self.assertEqual(sides["2026-11-04/doorbell/132907_session.jpg"], "val")


class Paste(unittest.TestCase):
    def test_paste_stays_on_floor_and_keeps_size(self):
        base = Image.new("RGB", (200, 100), (90, 90, 90))
        patch = Image.new("RGB", (20, 10), (200, 150, 100))
        out = train.paste(base, patch, random.Random(1))
        self.assertEqual(out.size, base.size)
        diff = np.abs(np.asarray(out, np.int16) - np.asarray(base, np.int16)).sum(axis=2)
        rows = np.nonzero(diff.any(axis=1))[0]
        self.assertTrue(len(rows) > 0)
        self.assertGreaterEqual(rows.min(), int(100 * train.FLOOR_TOP))

    def test_patch_taller_than_floor_still_pastes(self):
        base = Image.new("RGB", (200, 100), (90, 90, 90))
        patch = Image.new("RGB", (20, 60), (200, 150, 100))  # taller than the floor strip
        for seed in range(20):
            out = train.paste(base, patch, random.Random(seed))
            self.assertEqual(out.size, base.size)

    def test_night_base_gets_a_grey_patch(self):
        base = Image.new("RGB", (200, 100), (60, 60, 60))
        patch = Image.new("RGB", (30, 15), (200, 120, 40))
        out = np.asarray(train.paste(base, patch, random.Random(2)), np.int16)
        self.assertTrue((np.ptp(out, axis=2) <= 1).all())  # no colour introduced

    def test_to_square_does_not_crop(self):
        im = Image.new("RGB", (360, 200), (0, 0, 0))
        im.paste((255, 255, 255), (0, 0, 10, 200))  # white strip at the very left edge
        sq = np.asarray(train.to_square(im))
        self.assertEqual(sq.shape[:2], (train.CROP_PX, train.CROP_PX))
        self.assertGreater(sq[:, 0].mean(), 200)  # edge survived


if __name__ == "__main__":
    unittest.main()
