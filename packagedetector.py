#!/usr/bin/env python3
"""package-detector: poll Reolink cameras, look for a package, flip a
Home Assistant binary_sensor per camera.

One poll = one ffmpeg snapshot per camera + one YOLO11n ONNX inference
per snapshot + a per-camera present/absent state machine. Steady state
writes nothing to Home Assistant; only state *changes* do.

Heavy dependencies (onnxruntime, numpy, Pillow, requests) are imported
lazily inside the functions that need them, so `--help`, `--check`
config validation, and the unit tests run without them installed.

Usage (see README for the docker form):
    python packagedetector.py --check     # config + ffmpeg + cameras, no writes
    python packagedetector.py --dry-run   # one poll, print HA payload, write nothing
    python packagedetector.py             # one poll, write state + HA
    python packagedetector.py --report 7  # state + recent transitions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger("packagedetector")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_INPUT = 640  # YOLO11n ONNX input size (square, letterboxed)
# Camera names become part of a Home Assistant entity id
# (binary_sensor.package_<name>), which allows only these characters.
CAMERA_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_URL_CREDENTIALS_RE = re.compile(r"(?i)\b(rtsps?://)[^/\s@]*@")


def redact_url(text: str) -> str:
    """Hide user:password in any rtsp:// URL inside `text`. ffmpeg's error
    lines start with the input URL, and those lines reach the container log,
    which fleet's health check copies into GitHub issues."""
    return _URL_CREDENTIALS_RE.sub(r"\1***@", text)


class ConfigError(Exception):
    """Bad env value. Exit code 2, same convention as birdlisten."""


@dataclass
class Camera:
    name: str
    url: str


@dataclass
class Config:
    cameras: list[Camera]
    poll_interval_seconds: int = 15
    model_path: str = "/models/package.onnx"
    confidence: float = 0.5
    confirm_hits: int = 2
    clear_misses: int = 5
    zones: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    ha_url: str = ""
    ha_token: str = ""
    data_dir: str = "/data"
    snapshot_timeout_s: int = 15
    ha_refresh_seconds: int = 300


def parse_cameras(value: str) -> list[Camera]:
    """`name=url,name=url` — the same format as birdlisten's CAMERAS."""
    cameras = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ConfigError(f"CAMERAS entry has no '=': {redact_url(part)!r}")
        name, url = part.split("=", 1)
        name, url = name.strip(), url.strip()
        if not name or not url:
            raise ConfigError(f"CAMERAS entry is incomplete: {redact_url(part)!r}")
        if not CAMERA_NAME_RE.match(name):
            raise ConfigError(f"camera name {name!r} must be lowercase letters, digits or _ "
                              "(it becomes binary_sensor.package_<name> in Home Assistant)")
        if not url.startswith("rtsp://"):
            scheme = url.split("://", 1)[0] if "://" in url else url[:8]
            raise ConfigError(f"camera {name}: URL must be rtsp://, got {scheme!r}://...")
        cameras.append(Camera(name, url))
    if not cameras:
        raise ConfigError("CAMERAS is empty")
    names = [c.name for c in cameras]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate camera name in CAMERAS: {names}")
    return cameras


def parse_zones(value: str) -> dict[str, tuple[float, float, float, float]]:
    """`porch=0,0.5,1,1,garage=0,0.3,1,1` — x_min,y_min,x_max,y_max,
    fractions of the frame. Parsed as a token stream because a zone
    itself contains commas."""
    zones: dict[str, tuple[float, float, float, float]] = {}
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    i = 0
    while i < len(tokens):
        if "=" not in tokens[i]:
            raise ConfigError(f"ZONES: expected 'name=x,y,x,y', got {tokens[i]!r}")
        name, first = tokens[i].split("=", 1)
        nums = [first, *tokens[i + 1:i + 4]]
        if len(nums) != 4:
            raise ConfigError(f"ZONES: camera {name} needs 4 numbers, got {nums}")
        try:
            box = tuple(float(n) for n in nums)
        except ValueError:
            raise ConfigError(f"ZONES: camera {name} has a non-number: {nums}") from None
        if not all(0.0 <= n <= 1.0 for n in box):
            raise ConfigError(f"ZONES: camera {name} values must be 0..1: {box}")
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ConfigError(f"ZONES: camera {name} min must be < max: {box}")
        zones[name.strip()] = box  # type: ignore[assignment]
        i += 4
    return zones


def _env_int(env: dict, key: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(key, "")
    if raw == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from None
    if not lo <= val <= hi:
        raise ConfigError(f"{key} must be {lo}..{hi}, got {val}")
    return val


def _env_float(env: dict, key: str, default: float) -> float:
    raw = env.get(key, "")
    if raw == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from None
    if not 0.0 <= val <= 1.0:
        raise ConfigError(f"{key} must be 0..1, got {val}")
    return val


def load_config(env: dict | None = None, require_ha: bool = False) -> Config:
    env = dict(os.environ) if env is None else env
    cameras = parse_cameras(env.get("CAMERAS", ""))
    zones = parse_zones(env["ZONES"]) if env.get("ZONES") else {}
    for name in zones:
        if name not in {c.name for c in cameras}:
            raise ConfigError(f"ZONES names unknown camera: {name!r}")
    cfg = Config(
        cameras=cameras,
        poll_interval_seconds=_env_int(env, "POLL_INTERVAL_SECONDS", 15, 15, 900),
        model_path=env.get("MODEL_PATH", "/models/package.onnx"),
        confidence=_env_float(env, "CONFIDENCE", 0.5),
        confirm_hits=_env_int(env, "CONFIRM_HITS", 2, 1, 10),
        clear_misses=_env_int(env, "CLEAR_MISSES", 5, 1, 10),
        zones=zones,
        ha_url=env.get("HA_URL", "").rstrip("/"),
        ha_token=env.get("HA_TOKEN", ""),
        data_dir=env.get("DATA_DIR", "/data"),
        snapshot_timeout_s=_env_int(env, "SNAPSHOT_TIMEOUT_S", 15, 3, 120),
        ha_refresh_seconds=_env_int(env, "HA_REFRESH_SECONDS", 300, 60, 3600),
    )
    if require_ha and (not cfg.ha_url or not cfg.ha_token):
        raise ConfigError("live run needs HA_URL and HA_TOKEN (not needed for --check/--dry-run)")
    return cfg


# ---------------------------------------------------------------- zones

def zone_contains(zone: tuple[float, float, float, float] | None,
                  box: tuple[float, float, float, float]) -> bool:
    """True if the detection box's centre is inside the zone (or no zone)."""
    if zone is None:
        return True
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


# ---------------------------------------------------------------- state

def update_state(present: bool, hits: int, misses: int, detected: bool,
                 confirm_hits: int, clear_misses: int) -> tuple[bool, int, int, bool]:
    """Per-camera present/absent state machine.

    Returns (new_present, new_hits, new_misses, changed). A single stray
    frame must not flip the sensor either way: present needs
    `confirm_hits` consecutive detections, absent needs `clear_misses`
    consecutive misses.
    """
    if detected:
        hits, misses = hits + 1, 0
        new_present = True if hits >= confirm_hits else present
    else:
        misses, hits = misses + 1, 0
        new_present = False if misses >= clear_misses else present
    return new_present, hits, misses, new_present != present


# ---------------------------------------------------------------- capture

def capture_snapshot(url: str, timeout_s: int) -> tuple[str | None, str]:
    """One frame via ffmpeg from the RTSP sub-stream. Returns
    (temp JPEG path, error string). On success the caller deletes the
    file; on failure it is already gone. Errors never contain credentials."""
    if shutil.which("ffmpeg") is None:
        return None, "ffmpeg not found"
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-rtsp_transport", "tcp", "-i", url,
           "-frames:v", "1", "-q:v", "2", "-y", tmp.name]
    err = ""
    try:
        subprocess.run(cmd, timeout=timeout_s, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if os.path.getsize(tmp.name) == 0:
            err = "ffmpeg produced an empty frame"
    except subprocess.TimeoutExpired:
        err = f"ffmpeg timed out after {timeout_s}s"
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip().splitlines()
        err = "ffmpeg failed: " + redact_url(detail[-1] if detail else "unknown error")
    if err:
        _remove(tmp.name)
        return None, err
    return tmp.name, ""


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------- inference

def _nms(boxes: list, scores: list, iou_thresh: float = 0.45) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            x1 = max(boxes[i][0], boxes[j][0]); y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2]); y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            a = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            b = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            iou = inter / (a + b - inter) if (a + b - inter) > 0 else 0.0
            if iou <= iou_thresh:
                rest.append(j)
        order = rest
    return keep


def letterbox_params(orig_w: int, orig_h: int, size: int = MODEL_INPUT) -> tuple[float, int, int]:
    """(scale, pad_x, pad_y) used to fit the frame into a size x size canvas,
    centred, the way detect() and Ultralytics do it."""
    scale = min(size / orig_w, size / orig_h)
    nw, nh = int(orig_w * scale), int(orig_h * scale)
    return scale, (size - nw) // 2, (size - nh) // 2


def model_box_to_frame(cx: float, cy: float, w: float, h: float, orig_w: int, orig_h: int,
                       size: int = MODEL_INPUT) -> tuple[float, float, float, float]:
    """Map a (cx, cy, w, h) box in letterboxed-canvas pixels back to
    (x1, y1, x2, y2) as fractions of the ORIGINAL frame, clipped to 0..1.
    Without removing the padding, a 16:9 frame's top edge comes out at
    y~0.22 and its bottom at y~0.78, which breaks zones and evidence boxes."""
    scale, pad_x, pad_y = letterbox_params(orig_w, orig_h, size)

    def fx(v: float) -> float:
        return min(max((v - pad_x) / scale / orig_w, 0.0), 1.0)

    def fy(v: float) -> float:
        return min(max((v - pad_y) / scale / orig_h, 0.0), 1.0)

    return fx(cx - w / 2), fy(cy - h / 2), fx(cx + w / 2), fy(cy + h / 2)


_sessions: dict[tuple[str, float], object] = {}


def _session(model_path: str):
    """One ONNX Runtime session per model file, reused across cameras and
    polls (loop.py keeps this process alive). A new file mtime reloads it."""
    import onnxruntime as ort
    key = (model_path, os.path.getmtime(model_path))
    if key not in _sessions:
        _sessions.clear()
        _sessions[key] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _sessions[key]


def detect(image_path: str, model_path: str, confidence: float) -> list[dict]:
    """Run the ONNX model on one JPEG. Returns detections as
    {box: (x1,y1,x2,y2) normalised 0..1 of the original frame, confidence}.

    Assumes an Ultralytics-style export: one output, rows of
    (cx, cy, w, h, class scores...), single class `package`.
    TODO (issue #1): verify this against the actual exported model —
    output layout is the first thing to confirm with real weights,
    before trusting any detection.
    """
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        # Letterbox to square MODEL_INPUT, as Ultralytics does.
        scale, pad_x, pad_y = letterbox_params(orig_w, orig_h)
        nw, nh = int(orig_w * scale), int(orig_h * scale)
        canvas = Image.new("RGB", (MODEL_INPUT, MODEL_INPUT), (114, 114, 114))
        canvas.paste(im.resize((nw, nh)), (pad_x, pad_y))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, :, :, :]
    session = _session(model_path)
    out = session.run(None, {session.get_inputs()[0].name: arr})[0]
    out = np.asarray(out)
    if out.ndim == 3:
        out = out[0]  # (channels, anchors) -> transpose below if needed
        if out.shape[0] < out.shape[1]:
            out = out.T
    boxes, scores = [], []
    for row in out:
        if len(row) < 5:
            continue
        cx, cy, w, h = (float(v) for v in row[:4])
        score = float(np.max(row[4:]))
        if score < confidence:
            continue
        boxes.append(model_box_to_frame(cx, cy, w, h, orig_w, orig_h))
        scores.append(score)
    result = [{"box": boxes[i], "confidence": scores[i]} for i in _nms(boxes, scores)]
    return result


def annotate(image_path: str, detections: list[dict], dest_path: str) -> None:
    """Draw boxes on a copy for the evidence file. Best effort only."""
    try:
        from PIL import Image, ImageDraw
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            w, h = im.size
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline="red", width=3)
                draw.text((x1 * w + 2, y1 * h + 2), f"package {det['confidence']:.2f}", fill="red")
            im.save(dest_path, "JPEG", quality=85)
    except Exception:  # noqa: BLE001 — evidence must never break a poll
        log.warning("could not annotate evidence snapshot", exc_info=True)
        shutil.copyfile(image_path, dest_path)


# ---------------------------------------------------------------- storage

def init_db(data_dir: str) -> sqlite3.Connection:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "evidence"), exist_ok=True)
    conn = sqlite3.connect(os.path.join(data_dir, "package-detector.sqlite"))
    conn.execute("""CREATE TABLE IF NOT EXISTS polls (
        ts TEXT, camera TEXT, detected INTEGER, confidence REAL, present INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS camera_state (
        camera TEXT PRIMARY KEY, present INTEGER, hits INTEGER, misses INTEGER,
        last_seen TEXT, since TEXT)""")
    # Added later; older databases get them here. ha_synced: did the last
    # POST of this camera's state succeed. ha_posted: epoch of that POST.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(camera_state)")}
    if "ha_synced" not in cols:
        conn.execute("ALTER TABLE camera_state ADD COLUMN ha_synced INTEGER DEFAULT 0")
    if "ha_posted" not in cols:
        conn.execute("ALTER TABLE camera_state ADD COLUMN ha_posted REAL")
    conn.commit()
    return conn


@dataclass
class CamState:
    present: bool = False
    hits: int = 0
    misses: int = 0
    last_seen: str | None = None   # last poll with a detection
    since: str | None = None       # when `present` last changed
    ha_synced: bool = False
    ha_posted: float | None = None


def load_state(conn: sqlite3.Connection, camera: str) -> CamState:
    row = conn.execute("""SELECT present, hits, misses, last_seen, since, ha_synced, ha_posted
                          FROM camera_state WHERE camera=?""", (camera,)).fetchone()
    if not row:
        return CamState()
    return CamState(bool(row[0]), row[1], row[2], row[3], row[4], bool(row[5]), row[6])


def save_state(conn: sqlite3.Connection, camera: str, s: CamState) -> None:
    conn.execute("""INSERT OR REPLACE INTO camera_state
                    (camera, present, hits, misses, last_seen, since, ha_synced, ha_posted)
                    VALUES (?,?,?,?,?,?,?,?)""",
                 (camera, int(s.present), s.hits, s.misses, s.last_seen, s.since,
                  int(s.ha_synced), s.ha_posted))


def needs_ha_post(changed: bool, synced: bool, last_post: float | None, now: float,
                  refresh_s: int) -> bool:
    """POST on a state change, after a failed POST, and every `refresh_s`
    regardless. States set through HA's REST API vanish when HA restarts,
    and a failed POST would otherwise leave HA wrong until the next change."""
    return changed or not synced or last_post is None or now - last_post >= refresh_s


# ---------------------------------------------------------------- Home Assistant

def ha_payload(camera: str, present: bool, confidence: float | None,
               last_seen: str | None) -> dict:
    return {"state": "on" if present else "off",
            "attributes": {"friendly_name": f"Package {camera}",
                           "confidence": confidence,
                           "last_seen": last_seen,
                           "device_class": "occupancy"}}


def post_to_ha(cfg: Config, camera: str, present: bool,
               confidence: float | None, last_seen: str | None) -> str:
    """POST the sensor state. Returns "" on success, else an error string."""
    import requests  # lazy: see module docstring
    url = f"{cfg.ha_url}/api/states/binary_sensor.package_{camera}"
    try:
        resp = requests.post(url, json=ha_payload(camera, present, confidence, last_seen),
                             headers={"Authorization": f"Bearer {cfg.ha_token}"},
                             timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — HA failure must not kill the poll
        return f"HA POST failed: {exc}"
    return ""


# ---------------------------------------------------------------- poll

def poll_once(cfg: Config, dry_run: bool = False) -> int:
    """One poll over all cameras. Returns 0 on success, 1 if any camera failed.
    dry_run: capture + inference only; no db writes, no HA calls."""
    conn = None if dry_run else init_db(cfg.data_dir)
    now = datetime.now(timezone.utc).isoformat()
    rc = 0
    model_ok = os.path.exists(cfg.model_path)
    if not model_ok:
        log.warning("model not found at %s; running capture-only (no detections)",
                    cfg.model_path)
    for cam in cfg.cameras:
        snap, err = capture_snapshot(cam.url, cfg.snapshot_timeout_s)
        if snap is None:
            log.error("camera %s: %s", cam.name, err)
            rc = 1
            continue
        try:
            rc |= _poll_camera(cfg, conn, cam, snap, model_ok, now, dry_run)
        finally:
            # capture_snapshot used a temp file: remove it on every path.
            _remove(snap)
    if conn is not None:
        conn.close()
    return rc


def _poll_camera(cfg: Config, conn: sqlite3.Connection | None, cam: Camera, snap: str,
                 model_ok: bool, now: str, dry_run: bool) -> int:
    try:
        detections = detect(snap, cfg.model_path, cfg.confidence) if model_ok else []
    except Exception:  # noqa: BLE001 — one bad inference must not kill the poll
        log.error("camera %s: inference failed", cam.name, exc_info=True)
        return 1
    kept = [d for d in detections
            if zone_contains(cfg.zones.get(cam.name), d["box"])]
    detected = bool(kept)
    best = max((d["confidence"] for d in kept), default=None)
    log.info("camera %s: detected=%s best=%.2f kept=%d raw=%d",
             cam.name, detected, best or 0.0, len(kept), len(detections))
    if dry_run:
        if cfg.ha_url:
            log.info("dry-run HA payload for %s: %s",
                     cam.name, json.dumps(ha_payload(cam.name, detected, best, now)))
        return 0
    assert conn is not None
    s = load_state(conn, cam.name)
    new_present, s.hits, s.misses, changed = update_state(
        s.present, s.hits, s.misses, detected, cfg.confirm_hits, cfg.clear_misses)
    old_present, s.present = s.present, new_present
    if detected:
        s.last_seen = now
    if changed or s.since is None:
        s.since = now
    conn.execute("INSERT INTO polls VALUES (?,?,?,?,?)",
                 (now, cam.name, int(detected), best, int(new_present)))
    rc = 0
    ev = None
    if changed:
        day = datetime.now().strftime("%Y-%m-%d")
        ev_dir = os.path.join(cfg.data_dir, "evidence", day)
        os.makedirs(ev_dir, exist_ok=True)
        ev = os.path.join(ev_dir, f"{cam.name}_{int(time.time())}.jpg")
        annotate(snap, kept, ev)
    epoch = time.time()
    if needs_ha_post(changed, s.ha_synced, s.ha_posted, epoch, cfg.ha_refresh_seconds):
        err = post_to_ha(cfg, cam.name, new_present, best, s.last_seen)
        s.ha_synced = not err
        if err:
            log.error("camera %s: %s (will retry next poll)", cam.name, err)
            rc = 1
        else:
            s.ha_posted = epoch
            if changed:
                log.info("camera %s: state %s -> %s, HA updated, evidence %s",
                         cam.name, old_present, new_present, ev)
    save_state(conn, cam.name, s)
    conn.commit()
    return rc


def cmd_check(cfg: Config) -> int:
    """Config + ffmpeg + model presence + one snapshot per camera. No writes."""
    problems = []
    print(f"config: {len(cfg.cameras)} camera(s), poll {cfg.poll_interval_seconds}s, "
          f"confidence {cfg.confidence}, confirm {cfg.confirm_hits}, clear {cfg.clear_misses}")
    for cam in cfg.cameras:
        if "_main" in cam.url or "Preview_01_main" in cam.url or "h264Preview_01_main" in cam.url:
            print(f"hint: {cam.name} uses a main-stream URL; use the sub-stream")
    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg not found")
    if not os.path.exists(cfg.model_path):
        problems.append(f"model not found at {cfg.model_path} (capture-only until weights arrive)")
    for cam in cfg.cameras:
        snap, err = capture_snapshot(cam.url, cfg.snapshot_timeout_s)
        if snap is None:
            problems.append(f"camera {cam.name}: {err}")
            print(f"camera {cam.name}: FAIL {err}")
        else:
            print(f"camera {cam.name}: snapshot OK ({os.path.getsize(snap)} bytes)")
            os.unlink(snap)
    if problems:
        print("problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("check OK")
    return 0


def cmd_report(cfg: Config, days: int) -> int:
    db = os.path.join(cfg.data_dir, "package-detector.sqlite")
    if not os.path.exists(db):
        print("no polls yet (no database)")
        return 0
    conn = sqlite3.connect(db)
    for cam in cfg.cameras:
        row = conn.execute("SELECT present, last_seen, since FROM camera_state WHERE camera=?",
                           (cam.name,)).fetchone()
        if row is None:
            print(f"{cam.name}: no polls yet")
            continue
        print(f"{cam.name}: present={bool(row[0])} since={row[2]} last_seen={row[1]}")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="package-detector: poll cameras for packages")
    parser.add_argument("--check", action="store_true", help="config + cameras, no writes")
    parser.add_argument("--dry-run", action="store_true", help="one poll, no writes, no HA calls")
    parser.add_argument("--report", type=int, metavar="N", help="state summary (N unused, kept for birdlisten parity)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(require_ha=not (args.check or args.dry_run or args.report is not None))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.check:
        return cmd_check(cfg)
    if args.report is not None:
        return cmd_report(cfg, args.report)
    return poll_once(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
