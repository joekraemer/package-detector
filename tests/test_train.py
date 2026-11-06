"""Tests for train.py's labelling, split and paste helpers. No torch needed.
Run: python -m unittest discover -s tests
"""

import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402


def r(file, reason="baseline", tag=None, camera="doorbell"):
    t = file.rsplit("/", 1)[1][:6]  # HHMMSS from the file name, any camera
    row = {"file": file, "camera": camera, "reason": reason, "ts": f"{file[:10]}T{t[:2]}:{t[2:4]}:{t[4:6]}-08:00"}
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

    def test_uncertain_frames_wait_for_a_human(self):
        rows = [r("2026-11-02/doorbell/101500_uncertain.jpg", "uncertain"),
                r("2026-11-02/doorbell/111500_uncertain.jpg", "uncertain"),
                r("2026-11-02/doorbell/121500_during_3s.jpg", "during_3s")]
        got = train.resolve_labels(rows, {"2026-11-02/doorbell/111500_uncertain.jpg": "package"})
        self.assertNotIn("2026-11-02/doorbell/101500_uncertain.jpg", got)  # quiet day, still not "empty"
        self.assertEqual(got["2026-11-02/doorbell/111500_uncertain.jpg"], "package")
        self.assertEqual(got["2026-11-02/doorbell/121500_during_3s.jpg"], "empty")  # like any event frame


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


class Staged(unittest.TestCase):
    """Staging without session frames: hand labels marked staged."""

    GROWS = [
        r("2026-11-02/garage/100000_baseline.jpg", camera="garage"),
        r("2026-11-03/garage/100000_baseline.jpg", camera="garage"),
        r("2026-11-03/garage/110000_baseline.jpg", camera="garage"),
        r("2026-11-04/garage/100000_baseline.jpg", camera="garage"),   # staging day, unlabelled
        r("2026-11-04/garage/165838_after_30s.jpg", "after_30s", camera="garage"),
        r("2026-11-04/garage/170138_after_30s.jpg", "after_30s", camera="garage"),
        r("2026-11-04/garage/170206_baseline.jpg", camera="garage"),
    ]
    HUMAN = {"2026-11-04/garage/165838_after_30s.jpg": "package",
             "2026-11-04/garage/170138_after_30s.jpg": "package",
             "2026-11-04/garage/170206_baseline.jpg": "package"}

    def test_read_staged_days(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        p = d / "labels.jsonl"
        p.write_text(json.dumps({"file": "2026-11-04/garage/165838_after_30s.jpg", "label": "package",
                                 "staged": True}) + "\n"
                     + json.dumps({"file": "2026-11-03/doorbell/180533_arrive.jpg", "label": "package"}) + "\n")
        self.assertEqual(train.read_staged_days(p), {("garage", "2026-11-04")})

    def test_staged_day_is_not_auto_empty_and_not_the_val_day(self):
        staged = {("garage", "2026-11-04")}
        before = train.resolve_labels(self.GROWS, self.HUMAN)
        self.assertEqual(before["2026-11-04/garage/100000_baseline.jpg"], "empty")  # the old bug
        labels = train.resolve_labels(self.GROWS, self.HUMAN, staged)
        self.assertNotIn("2026-11-04/garage/100000_baseline.jpg", labels)
        sides = train.split(self.GROWS, labels, staged)
        # Val day is the newest quiet day (11-03), so the staged positives
        # are split by time instead of all landing in val with their day.
        self.assertEqual(sides["2026-11-03/garage/110000_baseline.jpg"], "val")
        self.assertEqual(sides["2026-11-04/garage/165838_after_30s.jpg"], "train")
        self.assertEqual(sides["2026-11-04/garage/170206_baseline.jpg"], "val")

    def test_group_split_puts_some_night_positives_in_train(self):
        rows = [r(f"2026-11-04/doorbell/{t}_session.jpg", "session", "box") for t in
                ("120000", "121000", "122000", "123000", "170000", "171000", "172000", "173000")]
        labels = train.resolve_labels(rows, {})
        night = lambda f: f[20:22] >= "17"  # noqa: E731
        plain = train.split(rows, labels)
        grouped = train.split(rows, labels, group=night)
        # Plain: the newest 30% overall is val, and all of it is night.
        self.assertEqual(sum(plain[f] == "val" for f in labels if not night(f)), 0)
        self.assertEqual(sum(grouped[f] == "train" for f in labels if night(f)), 3)
        self.assertEqual(sum(grouped[f] == "train" for f in labels if not night(f)), 3)


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
