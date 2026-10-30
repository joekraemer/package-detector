#!/usr/bin/env python3
"""capture.py: collect training frames for the package classifier.

Not part of the deployed detector. It runs for a couple of weeks on a
machine on the home network and fills CAPTURE_DIR with camera frames
plus an index, which autolabel/train (later) consume. See docs/MODEL.md.

Three ways frames get taken:

- baseline: every CAPTURE_DAY_MINUTES while the sun is up, every
  CAPTURE_NIGHT_MINUTES at night, and every CAPTURE_TWILIGHT_MINUTES
  within 45 min of sunrise/sunset (that is when Reolink cameras switch
  in and out of IR night mode, the frames the model most needs to see).
  Day/night comes from Home Assistant's sun.sun entity.
- event: when a camera's trigger sensor (e.g. the doorbell's person
  sensor) turns on, one "arrive" frame; when it turns off, one frame
  per CAPTURE_AFTER_SECONDS (default 30 s and 120 s later). The
  "after" frames are where a delivered package shows up.
- session: you stage packages and press Enter per shot; every frame is
  tagged with what is on the porch, so it labels itself.

Usage:
    python capture.py check                       # config, ffmpeg, HA entities, 1 frame/camera; keeps nothing
    python capture.py run                         # the long-running collector
    python capture.py session --tag empty         # interactive staged shots

Output, per local day:
    CAPTURE_DIR/YYYY-MM-DD/<camera>/<HHMMSS>_<reason>.jpg
    CAPTURE_DIR/YYYY-MM-DD/index.jsonl   one JSON object per frame
Logs and the index never contain camera passwords or the HA token.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import packagedetector as pd
from packagedetector import Camera, ConfigError, parse_cameras, redact_url

log = logging.getLogger("capture")

TWILIGHT_WINDOW = timedelta(minutes=45)
TRIGGER_POLL_S = 2.0
SUN_REFRESH_S = 300.0


# ---------------------------------------------------------------- config

@dataclass
class CaptureConfig:
    cameras: list[Camera]
    ha_url: str = ""
    ha_token: str = ""
    capture_dir: str = "package-data"
    triggers: dict[str, str] = field(default_factory=dict)  # camera -> entity_id
    day_minutes: int = 20
    night_minutes: int = 60
    twilight_minutes: int = 10
    after_seconds: tuple[int, ...] = (30, 120)
    max_events_per_hour: int = 12
    min_free_gb: float = 5.0
    snapshot_timeout_s: int = 15


def load_env_file(path: str, env: dict) -> None:
    """KEY=VALUE lines into `env`, without overriding keys already set.
    Splits on the first '=' only, because values such as CAMERAS
    (name=url,...) contain '=' themselves. Strips matching quotes."""
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            env.setdefault(key, value)


def parse_triggers(value: str, camera_names: list[str]) -> dict[str, str]:
    """`doorbell=binary_sensor.doorbell_person,garage=binary_sensor.garage_motion`."""
    triggers: dict[str, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ConfigError(f"CAPTURE_TRIGGERS entry has no '=': {part!r}")
        cam, entity = (s.strip() for s in part.split("=", 1))
        if cam not in camera_names:
            raise ConfigError(f"CAPTURE_TRIGGERS names unknown camera: {cam!r}")
        if not entity.startswith("binary_sensor."):
            raise ConfigError(f"CAPTURE_TRIGGERS {cam}: expected a binary_sensor.*, got {entity!r}")
        triggers[cam] = entity
    return triggers


def _int(env: dict, key: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(key, "")
    if raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from None
    if not lo <= v <= hi:
        raise ConfigError(f"{key} must be {lo}..{hi}, got {v}")
    return v


def load_capture_config(env: dict) -> CaptureConfig:
    if not env.get("CAMERAS"):
        raise ConfigError("CAMERAS is not set")
    cameras = parse_cameras(env["CAMERAS"])
    names = [c.name for c in cameras]
    after_raw = env.get("CAPTURE_AFTER_SECONDS", "30,120")
    try:
        after = tuple(sorted(int(x) for x in after_raw.split(",") if x.strip()))
    except ValueError:
        raise ConfigError(f"CAPTURE_AFTER_SECONDS must be integers, got {after_raw!r}") from None
    if not after or after[0] < 1 or after[-1] > 1800:
        raise ConfigError(f"CAPTURE_AFTER_SECONDS values must be 1..1800, got {after_raw!r}")
    try:
        min_free = float(env.get("CAPTURE_MIN_FREE_GB", "5"))
    except ValueError:
        raise ConfigError("CAPTURE_MIN_FREE_GB must be a number") from None
    return CaptureConfig(
        cameras=cameras,
        ha_url=env.get("HA_URL", "").rstrip("/"),
        ha_token=env.get("HA_TOKEN", ""),
        capture_dir=env.get("CAPTURE_DIR", "package-data"),
        triggers=parse_triggers(env.get("CAPTURE_TRIGGERS", ""), names),
        day_minutes=_int(env, "CAPTURE_DAY_MINUTES", 20, 1, 240),
        night_minutes=_int(env, "CAPTURE_NIGHT_MINUTES", 60, 1, 720),
        twilight_minutes=_int(env, "CAPTURE_TWILIGHT_MINUTES", 10, 1, 120),
        after_seconds=after,
        max_events_per_hour=_int(env, "CAPTURE_MAX_EVENTS_PER_HOUR", 12, 1, 120),
        min_free_gb=min_free,
        snapshot_timeout_s=_int(env, "SNAPSHOT_TIMEOUT_S", 15, 3, 120),
    )


# ---------------------------------------------------------------- sun phase

def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def sun_phase(sun: dict | None, now: datetime) -> str:
    """'day', 'night' or 'twilight' from an HA sun.sun state object.
    Unknown (HA down, no sun entity) counts as 'day': over-capturing a
    little beats missing deliveries. `now` must be timezone-aware."""
    if not sun or sun.get("state") not in ("above_horizon", "below_horizon"):
        return "day"
    attrs = sun.get("attributes", {})
    marks = [_parse_iso(attrs.get("next_rising")), _parse_iso(attrs.get("next_setting")),
             _parse_iso(sun.get("last_changed"))]
    if any(m is not None and abs(now - m) <= TWILIGHT_WINDOW for m in marks):
        return "twilight"
    return "day" if sun["state"] == "above_horizon" else "night"


def baseline_interval_s(cfg: CaptureConfig, phase: str) -> float:
    minutes = {"day": cfg.day_minutes, "night": cfg.night_minutes,
               "twilight": cfg.twilight_minutes}[phase]
    return minutes * 60.0


def baseline_due(cfg: CaptureConfig, last: float | None, now: float, phase: str) -> bool:
    """Judged against the CURRENT phase's interval, not the one in force
    at the last shot: a night frame at 05:44 must not push the next frame
    to 06:44 when twilight (10 min cadence) starts at 05:45."""
    return last is None or now - last >= baseline_interval_s(cfg, phase)


# ---------------------------------------------------------------- triggers

@dataclass
class Shot:
    due: float
    camera: str
    reason: str
    event: str


class TriggerTracker:
    """Turns trigger-sensor states into planned shots. Pure: no I/O.

    on  (from off)  -> an 'arrive' shot now, and any pending 'after'
                       shots for that camera are dropped (the person
                       came back; the visit is not over yet).
    off (from on)   -> one 'after_<N>s' shot per CAPTURE_AFTER_SECONDS.
    unavailable/unknown/None leave the remembered state alone, so a
    sensor blipping through 'unavailable' does not fake a visit.
    The first state seen for a camera is only remembered, never acted
    on: starting up mid-visit is not an arrival.
    Arrivals are capped per camera per rolling hour (a garage motion
    sensor in a windstorm would otherwise fill the disk); the 'after'
    shots of a capped arrival are skipped too."""

    def __init__(self, after_seconds: tuple[int, ...], max_per_hour: int):
        self.after_seconds = after_seconds
        self.max_per_hour = max_per_hour
        self.state: dict[str, str] = {}
        self.arrivals: dict[str, deque] = {}
        self.event_id: dict[str, str | None] = {}
        self.pending: list[Shot] = []

    def update(self, camera: str, new: str | None, now: float) -> list[Shot]:
        """Feed one observed state; returns shots due immediately."""
        if new not in ("on", "off"):
            return []
        old = self.state.get(camera)
        self.state[camera] = new
        if old is None or old == new:
            return []
        if new == "on":
            self.pending = [s for s in self.pending if s.camera != camera]
            hist = self.arrivals.setdefault(camera, deque())
            while hist and now - hist[0] > 3600:
                hist.popleft()
            if len(hist) >= self.max_per_hour:
                self.event_id[camera] = None
                log.info("%s: trigger on, but %d events in the last hour; not capturing",
                         camera, len(hist))
                return []
            hist.append(now)
            eid = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
            self.event_id[camera] = eid
            return [Shot(now, camera, "arrive", eid)]
        eid = self.event_id.get(camera)
        if eid is None:
            return []
        for s in self.after_seconds:
            self.pending.append(Shot(now + s, camera, f"after_{s}s", eid))
        return []

    def due(self, now: float) -> list[Shot]:
        ready = [s for s in self.pending if s.due <= now]
        self.pending = [s for s in self.pending if s.due > now]
        return sorted(ready, key=lambda s: s.due)


# ---------------------------------------------------------------- Home Assistant

class HomeAssistant:
    """Tiny REST client. Failures are logged at most once per entity per
    WARN_EVERY_S, plus one line when the entity is reachable again: the
    trigger poll runs every 2 s, and fleet's health check opens an issue
    for any log line matching 'error', so a HA restart must not produce
    a flood of lines, nor an exception class name like ConnectionError."""

    WARN_EVERY_S = 1800.0

    def __init__(self, url: str, token: str):
        self.url, self.token = url, token
        self._failing_since: dict[str, float] = {}
        self._last_warn: dict[str, float] = {}

    @property
    def configured(self) -> bool:
        return bool(self.url and self.token)

    def _fail(self, entity_id: str, why: str, now: float) -> None:
        self._failing_since.setdefault(entity_id, now)
        if now - self._last_warn.get(entity_id, -1e18) >= self.WARN_EVERY_S:
            mins = (now - self._failing_since[entity_id]) / 60
            log.warning("HA %s unavailable (%s) for %.0f min; retrying every %.0f s",
                        entity_id, why, mins, TRIGGER_POLL_S)
            self._last_warn[entity_id] = now

    def _ok(self, entity_id: str, now: float) -> None:
        since = self._failing_since.pop(entity_id, None)
        if since is not None:
            self._last_warn.pop(entity_id, None)
            log.info("HA %s reachable again after %.0f min", entity_id, (now - since) / 60)

    def state(self, entity_id: str) -> dict | None:
        """GET /api/states/<id>. None on any failure (logged, token-free)."""
        import requests
        now = time.time()
        try:
            r = requests.get(f"{self.url}/api/states/{entity_id}", timeout=5,
                             headers={"Authorization": f"Bearer {self.token}"})
        except requests.RequestException:
            self._fail(entity_id, "unreachable", now)
            return None
        if r.status_code != 200:
            self._fail(entity_id, f"HTTP {r.status_code}", now)
            return None
        self._ok(entity_id, now)
        return r.json()


# ---------------------------------------------------------------- saving frames

def free_gb(path: str) -> float:
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def take(cfg: CaptureConfig, camera: Camera, reason: str, phase: str,
         event: str | None = None, tag: str | None = None,
         when: datetime | None = None) -> str | None:
    """One frame, saved and indexed. Returns the saved path or None."""
    if free_gb(cfg.capture_dir) < cfg.min_free_gb:
        log.warning("less than %.1f GB free in %s; not capturing", cfg.min_free_gb, cfg.capture_dir)
        return None
    tmp, err = pd.capture_snapshot(camera.url, cfg.snapshot_timeout_s)
    if tmp is None:
        log.warning("%s: %s", camera.name, err)
        return None
    when = when or datetime.now().astimezone()
    day = when.strftime("%Y-%m-%d")
    rel = os.path.join(day, camera.name, f"{when.strftime('%H%M%S')}_{reason}.jpg")
    dest = os.path.join(cfg.capture_dir, rel)
    n = 1
    while os.path.exists(dest):  # two shots in the same second (session mode)
        n += 1
        rel = os.path.join(day, camera.name, f"{when.strftime('%H%M%S')}_{reason}_{n}.jpg")
        dest = os.path.join(cfg.capture_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)
    record = {"file": rel, "camera": camera.name, "ts": when.isoformat(timespec="seconds"),
              "reason": reason, "phase": phase}
    if event:
        record["event"] = event
    if tag:
        record["tag"] = tag
    with open(os.path.join(cfg.capture_dir, day, "index.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    log.info("%s %s -> %s", camera.name, reason, rel)
    return dest


# ---------------------------------------------------------------- modes

def cmd_check(cfg: CaptureConfig) -> int:
    ok = True
    print(f"cameras: {', '.join(c.name for c in cfg.cameras)}")
    print(f"capture dir: {os.path.abspath(cfg.capture_dir)} ({free_gb(cfg.capture_dir):.0f} GB free)")
    if shutil.which("ffmpeg") is None:
        print("FAIL ffmpeg not found on PATH")
        return 1
    ha = HomeAssistant(cfg.ha_url, cfg.ha_token)
    if not ha.configured:
        print("WARN HA_URL/HA_TOKEN not set: no event capture, baseline runs on the day interval")
    else:
        sun = ha.state("sun.sun")
        print(f"sun.sun: {sun['state'] if sun else 'UNAVAILABLE'} -> phase now: "
              f"{sun_phase(sun, datetime.now().astimezone())}")
        for cam, entity in cfg.triggers.items():
            st = ha.state(entity)
            if st is None:
                print(f"FAIL trigger {cam}: {entity} not found in HA")
                ok = False
            else:
                name = st.get("attributes", {}).get("friendly_name", "")
                print(f"trigger {cam}: {entity} ({name}) = {st.get('state')}")
    for cam in cfg.cameras:
        if cam.name not in cfg.triggers:
            print(f"note {cam.name}: no trigger configured, baseline frames only")
        tmp, err = pd.capture_snapshot(cam.url, cfg.snapshot_timeout_s)
        if tmp is None:
            print(f"FAIL {cam.name}: {err}")
            ok = False
        else:
            print(f"ok {cam.name}: frame {os.path.getsize(tmp) // 1024} KB")
            os.unlink(tmp)
    return 0 if ok else 1


def cmd_run(cfg: CaptureConfig) -> int:
    ha = HomeAssistant(cfg.ha_url, cfg.ha_token)
    if not ha.configured:
        log.warning("HA_URL/HA_TOKEN not set: baseline only, day interval")
    tracker = TriggerTracker(cfg.after_seconds, cfg.max_events_per_hour)
    by_name = {c.name: c for c in cfg.cameras}
    sun, sun_at = None, 0.0
    last_baseline: float | None = None
    log.info("capturing %s into %s", ", ".join(by_name), os.path.abspath(cfg.capture_dir))
    while True:
        now = time.time()
        if ha.configured and now - sun_at >= SUN_REFRESH_S:
            sun, sun_at = ha.state("sun.sun"), now
        phase = sun_phase(sun, datetime.now().astimezone())

        shots: list[Shot] = []
        if ha.configured:
            for cam, entity in cfg.triggers.items():
                st = ha.state(entity)
                shots += tracker.update(cam, st.get("state") if st else None, now)
        shots += tracker.due(now)
        for s in shots:
            take(cfg, by_name[s.camera], s.reason, phase, event=s.event)

        if baseline_due(cfg, last_baseline, now, phase):
            for cam in cfg.cameras:
                take(cfg, cam, "baseline", phase)
            last_baseline = now
        time.sleep(TRIGGER_POLL_S)


def cmd_session(cfg: CaptureConfig, tag: str, only: list[str] | None) -> int:
    cams = [c for c in cfg.cameras if not only or c.name in only]
    if not cams:
        print(f"no such camera: {only}")
        return 2
    print("Stage the porch, then:\n"
          "  Enter        shoot every selected camera with the current tag\n"
          "  t <new tag>  change the tag (use 'empty' for an empty-porch shot)\n"
          "  q            quit\n"
          "Tags: what is in the zone, e.g. brown-box-door, white-mailer-mat, 2-boxes-stacked.")
    shots = 0
    while True:
        try:
            line = input(f"[{tag}] > ").strip()
        except EOFError:
            break
        if line == "q":
            break
        if line.startswith("t "):
            tag = line[2:].strip() or tag
            continue
        if line:
            print("? Enter, 't <tag>' or 'q'")
            continue
        for cam in cams:
            if take(cfg, cam, "session", "session", tag=tag):
                shots += 1
    print(f"{shots} frames saved under {os.path.abspath(cfg.capture_dir)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", default=".env",
                    help="KEY=VALUE file read if present (real environment wins); default .env")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("check")
    sub.add_parser("run")
    sp = sub.add_parser("session")
    sp.add_argument("--tag", required=True, help="what is staged in the zone, or 'empty'")
    sp.add_argument("--camera", action="append", help="limit to this camera (repeatable)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    env = dict(os.environ)
    if os.path.exists(args.env_file):
        load_env_file(args.env_file, env)
    try:
        cfg = load_capture_config(env)
    except ConfigError as exc:
        print(f"config error: {redact_url(str(exc))}", file=sys.stderr)
        return 2
    if args.mode == "check":
        return cmd_check(cfg)
    if args.mode == "session":
        return cmd_session(cfg, args.tag, args.camera)
    try:
        return cmd_run(cfg)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
