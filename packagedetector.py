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
import io
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
import urllib.parse
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger("packagedetector")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_INPUT = 640  # YOLO11n ONNX input size (square, letterboxed)
# Zone classifier (train.py): crops are saved at CLS_CROP_PX square, then
# the network resizes them to the model's imgsz. classify() repeats both
# steps so the service sees exactly what training and evaluation saw.
CLS_CROP_PX = 256
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
    # The push bar: an arrival notifies only once CONFIRM_HITS consecutive
    # polls score at least this. The sensor, evidence and uncertain frames
    # still follow `confidence`. Defaults to `confidence` (no extra gate).
    notify_confidence: float = 0.5
    confirm_hits: int = 2
    clear_misses: int = 5
    zones: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    ha_url: str = ""
    ha_token: str = ""
    data_dir: str = "/data"
    snapshot_timeout_s: int = 15
    ha_refresh_seconds: int = 300
    # Shadow mode: run the full loop (state machine, SQLite, evidence) but
    # never call Home Assistant; state changes are only logged.
    ha_dry_run: bool = False
    # Home Assistant notify services to call when a package ARRIVES, e.g.
    # ["notify.mobile_app_joes_iphone"]. Empty = sensor only, no pushes.
    ha_notify: list[str] = field(default_factory=list)
    # A camera that notified less than this long ago stays quiet on its next
    # arrival, so a package flickering in and out cannot spam the phone.
    ha_notify_cooldown_seconds: int = 1800
    # Any short value. Changing it re-sends the "Package alerts are on" push
    # once, to test the notification path without a package.
    ha_notify_test: str = ""
    # Attach the camera frame to each push: uploaded to HA's media folder
    # (needs an admin HA_TOKEN), sent as data.image. A failed upload still
    # sends the push, just without the photo.
    ha_notify_image: bool = True
    # Camera name -> HA camera entity, e.g. {"doorbell": "camera.doorbell_fluent"}.
    # Tapping that camera's push opens the entity's live view in the HA app.
    camera_entities: dict[str, str] = field(default_factory=dict)
    # "ffmpeg" (RTSP grab) or "reolink" (Snap API, ffmpeg as fallback).
    snapshot_source: str = "ffmpeg"
    # Frames the classifier is unsure about are the most useful training
    # data: polls with uncertain_low <= p < uncertain_high are saved raw
    # under DATA_DIR/frames, at most one per camera per uncertain_every_s.
    # uncertain_every_s = 0 turns it off.
    uncertain_low: float = 0.4
    uncertain_high: float = 0.8
    uncertain_every_s: int = 600


def _env_source(env: dict) -> str:
    raw = env.get("SNAPSHOT_SOURCE", "").strip().lower() or "ffmpeg"
    if raw not in SNAPSHOT_SOURCES:
        raise ConfigError(f"SNAPSHOT_SOURCE must be one of {SNAPSHOT_SOURCES}, got {raw!r}")
    return raw


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


def _env_bool(env: dict, key: str, default: bool = False) -> bool:
    raw = env.get(key, "").strip().lower()
    if raw == "":
        return default
    if raw in ("0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    raise ConfigError(f"{key} must be 1/0/true/false, got {raw!r}")


NOTIFY_RE = re.compile(r"^notify\.[a-z0-9_]+$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _env_token(env: dict, key: str) -> str:
    raw = env.get(key, "").strip()
    if raw and not TOKEN_RE.match(raw):
        raise ConfigError(f"{key} must be 1-64 letters, digits, '.', '_' or '-', got {raw!r}")
    return raw


def parse_notify(value: str) -> list[str]:
    """`notify.mobile_app_a,notify.mobile_app_b`; a bare `mobile_app_a` gets
    the `notify.` prefix. Becomes part of an HA API path, so it is strict."""
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "." not in part:
            part = f"notify.{part}"
        if not NOTIFY_RE.match(part):
            raise ConfigError(f"HA_NOTIFY entry must look like notify.mobile_app_xxx, got {part!r}")
        out.append(part)
    return out


CAMERA_ENTITY_RE = re.compile(r"^camera\.[a-z0-9_]+$")


def parse_camera_entities(value: str, cameras: list[Camera]) -> dict[str, str]:
    """`doorbell=camera.doorbell_fluent,porch=camera.porch`. Goes into a
    URL the phone opens, so entity ids are strict."""
    names = {c.name for c in cameras}
    out: dict[str, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, entity = (p.strip() for p in part.partition("="))
        if name not in names:
            raise ConfigError(f"CAMERA_ENTITIES names unknown camera: {name!r}")
        if not CAMERA_ENTITY_RE.match(entity):
            raise ConfigError(f"CAMERA_ENTITIES: {name} must map to camera.xxx, got {entity!r}")
        out[name] = entity
    return out


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
        notify_confidence=_env_float(env, "NOTIFY_CONFIDENCE", _env_float(env, "CONFIDENCE", 0.5)),
        confirm_hits=_env_int(env, "CONFIRM_HITS", 2, 1, 10),
        clear_misses=_env_int(env, "CLEAR_MISSES", 5, 1, 10),
        zones=zones,
        ha_url=env.get("HA_URL", "").rstrip("/"),
        ha_token=env.get("HA_TOKEN", ""),
        data_dir=env.get("DATA_DIR", "/data"),
        snapshot_timeout_s=_env_int(env, "SNAPSHOT_TIMEOUT_S", 15, 3, 120),
        ha_refresh_seconds=_env_int(env, "HA_REFRESH_SECONDS", 300, 60, 3600),
        ha_dry_run=_env_bool(env, "HA_DRY_RUN"),
        ha_notify=parse_notify(env.get("HA_NOTIFY", "")),
        ha_notify_cooldown_seconds=_env_int(env, "HA_NOTIFY_COOLDOWN_SECONDS", 1800, 0, 86400),
        ha_notify_test=_env_token(env, "HA_NOTIFY_TEST"),
        ha_notify_image=_env_bool(env, "HA_NOTIFY_IMAGE", default=True),
        camera_entities=parse_camera_entities(env.get("CAMERA_ENTITIES", ""), cameras),
        snapshot_source=_env_source(env),
        uncertain_low=_env_float(env, "UNCERTAIN_LOW", 0.4),
        uncertain_high=_env_float(env, "UNCERTAIN_HIGH", 0.8),
        uncertain_every_s=_env_int(env, "UNCERTAIN_EVERY_SECONDS", 600, 0, 86400),
    )
    if cfg.uncertain_low >= cfg.uncertain_high:
        raise ConfigError(f"UNCERTAIN_LOW ({cfg.uncertain_low}) must be < "
                          f"UNCERTAIN_HIGH ({cfg.uncertain_high})")
    if require_ha and not cfg.ha_dry_run and (not cfg.ha_url or not cfg.ha_token):
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

SNAPSHOT_SOURCES = ("ffmpeg", "reolink")
SNAP_WIDTH = 640  # the sub-stream width every training frame so far has
_REOLINK_CHANNEL_RE = re.compile(r"Preview_(\d+)_")


def reolink_snap_request(url: str) -> tuple[str, dict]:
    """The Reolink HTTP API request equivalent to an RTSP camera URL:
    same host and login, channel from `Preview_NN_` (NN is 1-based).
    Raises ValueError if the URL does not name a channel."""
    u = urllib.parse.urlsplit(url)
    m = _REOLINK_CHANNEL_RE.search(u.path)
    if not u.hostname or not m:
        raise ValueError("RTSP URL has no Reolink Preview_NN channel")
    params = {"cmd": "Snap", "channel": int(m.group(1)) - 1, "rs": f"{time.time():.3f}",
              "user": urllib.parse.unquote(u.username or ""),
              "password": urllib.parse.unquote(u.password or "")}
    return f"https://{u.hostname}/cgi-bin/api.cgi", params


def reolink_snapshot(url: str, timeout_s: int) -> tuple[str | None, str]:
    """One frame from the Reolink NVR/camera's Snap API: ~1 s, against ~4 s
    for an ffmpeg RTSP grab (which has to wait for a keyframe). The JPEG
    is main-stream resolution; it is downscaled to SNAP_WIDTH so frames
    keep the size the model was trained on. TLS is not verified: Reolink
    devices only have a self-signed certificate, and this runs on the LAN.
    The login travels as query parameters (Reolink's API has no other
    one-shot form), so errors report the exception CLASS only: requests
    puts the full URL, password included, in its messages."""
    import requests  # lazy: see module docstring
    from PIL import Image
    try:
        endpoint, params = reolink_snap_request(url)
    except ValueError as exc:
        return None, f"reolink snapshot: {exc}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # InsecureRequestWarning, see above
            resp = requests.get(endpoint, params=params, timeout=timeout_s, verify=False)
    except requests.RequestException as exc:
        return None, f"reolink snapshot failed: {type(exc).__name__}"
    if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("image/"):
        return None, f"reolink snapshot failed: HTTP {resp.status_code}, not an image"
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        with Image.open(io.BytesIO(resp.content)) as im:
            im = im.convert("RGB")
            if im.width > SNAP_WIDTH:
                im = im.resize((SNAP_WIDTH, round(im.height * SNAP_WIDTH / im.width)), Image.BOX)
            im.save(tmp.name, quality=92)
    except Exception as exc:  # noqa: BLE001 — a bad JPEG is one failed frame
        _remove(tmp.name)
        return None, f"reolink snapshot: unreadable image ({type(exc).__name__})"
    return tmp.name, ""


def capture_snapshot(url: str, timeout_s: int, source: str = "ffmpeg") -> tuple[str | None, str]:
    """One frame. Returns (temp JPEG path, error string). On success the
    caller deletes the file; on failure it is already gone. Errors never
    contain credentials. source="reolink" tries the Snap API first and
    falls back to ffmpeg, so a firmware or login change costs speed, not
    frames."""
    if source == "reolink":
        snap, err = reolink_snapshot(url, timeout_s)
        if snap is not None:
            return snap, ""
        _warn_snap_fallback(url, err)
    return _ffmpeg_snapshot(url, timeout_s)


_snap_fallback_warned: set[str] = set()


def _warn_snap_fallback(url: str, err: str) -> None:
    key = urllib.parse.urlsplit(url).path
    if key not in _snap_fallback_warned:  # once per camera, not every poll
        _snap_fallback_warned.add(key)
        log.warning("%s; falling back to ffmpeg", err)


def _ffmpeg_snapshot(url: str, timeout_s: int) -> tuple[str | None, str]:
    """One frame via ffmpeg from the RTSP sub-stream."""
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


def load_model_meta(model_path: str) -> dict | None:
    """The sidecar `<model>.json` that train.py writes next to a zone
    classifier. None means the model is the (older) box detector.

    {"classes": ["empty", "package"], "imgsz": 224,
     "cameras": {"doorbell": [x0, y0, x1, y1]}}   # zones it was trained on
    """
    path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        meta = json.load(fh)
    classes, cams = meta.get("classes"), meta.get("cameras")
    if (not isinstance(classes, list) or "package" not in classes
            or not isinstance(meta.get("imgsz"), int) or not isinstance(cams, dict)):
        raise ConfigError(f"{path}: expected classes (with 'package'), imgsz and cameras")
    return meta


def classifier_input(im, zone: tuple[float, float, float, float], size: int):
    """Zone crop -> CLS_CROP_PX square -> size square, RGB/255, NCHW float32.
    Must stay in step with train.py (zone_crop + to_square + the network's
    own resize); tests/test_packagedetector.py checks that it does."""
    import numpy as np
    from PIL import Image

    im = im.convert("RGB")
    w, h = im.size
    crop = im.crop((int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)))
    sq = crop.resize((CLS_CROP_PX, CLS_CROP_PX), Image.BILINEAR)
    x = np.asarray(sq.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None, :, :, :]


def classify(image_path: str, model_path: str, zone: tuple[float, float, float, float],
             meta: dict) -> float:
    """Probability that a package is in `zone` of this frame."""
    from PIL import Image

    with Image.open(image_path) as im:
        x = classifier_input(im, zone, meta["imgsz"])
    session = _session(model_path)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    return float(out.reshape(-1)[meta["classes"].index("package")])


_warned_no_zone: set[str] = set()


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
    # notify_pending: an arrival whose notification has not gone out yet.
    # notified: epoch of the last notification sent for this camera.
    if "notify_pending" not in cols:
        conn.execute("ALTER TABLE camera_state ADD COLUMN notify_pending INTEGER DEFAULT 0")
    if "notified" not in cols:
        conn.execute("ALTER TABLE camera_state ADD COLUMN notified REAL")
    # strong_hits: consecutive polls at or above NOTIFY_CONFIDENCE.
    if "strong_hits" not in cols:
        conn.execute("ALTER TABLE camera_state ADD COLUMN strong_hits INTEGER DEFAULT 0")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
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
    notify_pending: bool = False
    notified: float | None = None
    strong_hits: int = 0           # consecutive polls >= notify_confidence


def load_state(conn: sqlite3.Connection, camera: str) -> CamState:
    row = conn.execute("""SELECT present, hits, misses, last_seen, since, ha_synced, ha_posted,
                                 notify_pending, notified, strong_hits
                          FROM camera_state WHERE camera=?""", (camera,)).fetchone()
    if not row:
        return CamState()
    return CamState(bool(row[0]), row[1], row[2], row[3], row[4], bool(row[5]), row[6],
                    bool(row[7]), row[8], row[9] or 0)


def save_state(conn: sqlite3.Connection, camera: str, s: CamState) -> None:
    conn.execute("""INSERT OR REPLACE INTO camera_state
                    (camera, present, hits, misses, last_seen, since, ha_synced, ha_posted,
                     notify_pending, notified, strong_hits)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                 (camera, int(s.present), s.hits, s.misses, s.last_seen, s.since,
                  int(s.ha_synced), s.ha_posted, int(s.notify_pending), s.notified,
                  s.strong_hits))


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


def _tap_to_camera(data: dict, entity: str | None) -> None:
    """Tapping the push opens the camera's more-info view (live stream) in
    the HA app: `url` on iOS, `clickAction` on Android."""
    if entity:
        data["url"] = data["clickAction"] = f"/?more-info-entity-id={entity}"


def notify_payload(camera: str, confidence: float | None, when: datetime | None = None,
                   image: str | None = None, camera_entity: str | None = None) -> dict:
    when = when or datetime.now()
    at = when.strftime("%I:%M %p").lstrip("0")
    place = camera.replace("_", " ")
    conf = f" (confidence {confidence:.2f})" if confidence is not None else ""
    # `tag` makes a newer alert for the same camera replace the older one.
    data: dict = {"tag": f"package-{camera}"}
    if image:
        data["image"] = image
    _tap_to_camera(data, camera_entity)
    return {"title": f"Package at the {place}",
            "message": f"A package showed up at the {place} at {at}{conf}.",
            "data": data}


# Pushed frames go to HA's local media folder, /media/<NOTIFY_MEDIA_DIR>/
# on the HA host. The companion app fetches /media/local/... with its own
# login, so the photo is never public (unlike /config/www).
NOTIFY_MEDIA_DIR = "package-detector"


def upload_notify_image(cfg: Config, snap: str, camera: str,
                        when: datetime | None = None) -> tuple[str | None, str]:
    """Upload one frame to HA's media source. Returns (the relative URL a
    notification's data.image takes, "") or (None, error string). HA only
    accepts uploads from an admin user's token."""
    import requests  # lazy: see module docstring
    when = when or datetime.now()
    name = f"{camera}_{when.strftime('%Y%m%d_%H%M%S')}.jpg"
    try:
        with open(snap, "rb") as fh:
            resp = requests.post(
                f"{cfg.ha_url}/api/media_source/local_source/upload",
                data={"media_content_id": f"media-source://media_source/local/{NOTIFY_MEDIA_DIR}"},
                files={"file": (name, fh, "image/jpeg")},
                headers={"Authorization": f"Bearer {cfg.ha_token}"},
                timeout=15)
        resp.raise_for_status()
        content_id = resp.json()["media_content_id"]
    except Exception as exc:  # noqa: BLE001 — a photo must never cost the push
        hint = " (HA_TOKEN must belong to an admin user)" if "401" in str(exc) else ""
        return None, f"image upload failed: {exc}{hint}"
    prefix = "media-source://media_source/local/"
    if not content_id.startswith(prefix):
        return None, f"image upload returned an unexpected id: {content_id!r}"
    return "/media/local/" + content_id[len(prefix):], ""


def _notify_image(cfg: Config, snap: str | None, camera: str) -> str | None:
    if not (cfg.ha_notify_image and snap):
        return None
    image, err = upload_notify_image(cfg, snap, camera)
    if err:
        log.warning("camera %s: %s; sending the push without a photo", camera, err)
    return image


def notify_ha(cfg: Config, payload: dict) -> str:
    """Call every HA_NOTIFY service. Returns "" if all succeeded, else an
    error string naming the ones that failed (never the token)."""
    import requests  # lazy: see module docstring
    failed = []
    for target in cfg.ha_notify:
        domain, service = target.split(".", 1)
        try:
            resp = requests.post(f"{cfg.ha_url}/api/services/{domain}/{service}", json=payload,
                                 headers={"Authorization": f"Bearer {cfg.ha_token}"},
                                 timeout=10)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — a push failure must not kill the poll
            failed.append(f"{target}: {exc}")
    return "; ".join(failed)


def _notify_hello(cfg: Config, conn: sqlite3.Connection, snap: str | None = None,
                  camera: str | None = None) -> int:
    """One "alerts are on" push per HA_NOTIFY setting, so whoever turned
    alerts on gets proof on the phone that the path works. Stored in the
    db: restarts and image updates stay quiet; changing HA_NOTIFY says hello
    again. A failure is logged and retried on the next poll. With `snap`
    it carries that camera's current frame, which tests the photo path."""
    want = ",".join(cfg.ha_notify)
    if cfg.ha_notify_test:
        # Changing HA_NOTIFY_TEST (e.g. to today's date) sends it once more.
        want += f"|test={cfg.ha_notify_test}"
    row = conn.execute("SELECT value FROM meta WHERE key='notify_hello'").fetchone()
    if row and row[0] == want:
        return 0
    cams = ", ".join(c.name.replace("_", " ") for c in cfg.cameras)
    data: dict = {"tag": "package-detector-hello"}
    image = _notify_image(cfg, snap, camera or "hello")
    if image:
        data["image"] = image
    _tap_to_camera(data, cfg.camera_entities.get(camera or ""))
    err = notify_ha(cfg, {"title": "Package alerts are on",
                          "message": f"You'll get a notification when a package shows up ({cams}).",
                          "data": data})
    if err:
        log.error("HA notify (alerts-on message) failed: %s (will retry next poll)", err)
        return 1
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('notify_hello', ?)", (want,))
    conn.commit()
    log.info("HA notify: alerts-on message sent to %s", want)
    return 0


def _handle_notify(cfg: Config, cam: Camera, s: CamState, changed: bool, present: bool,
                   confidence: float | None, epoch: float, snap: str | None = None) -> int:
    """Arrival -> one notification, with this poll's frame attached. Leaving
    clears a notification that has not gone out yet; a failed one is retried
    while the package is there (with that later poll's frame). The push waits
    until CONFIRM_HITS consecutive polls reach NOTIFY_CONFIDENCE; an arrival
    that never does stays a sensor-only state change and is never pushed."""
    if changed and present:
        s.notify_pending = True
    if not present:
        s.notify_pending = False
    if not (s.notify_pending and cfg.ha_notify):
        return 0
    if s.strong_hits < cfg.confirm_hits:
        if changed:
            log.info("camera %s: present, but below NOTIFY_CONFIDENCE %.2f; push waits",
                     cam.name, cfg.notify_confidence)
        return 0
    if (s.notified is not None and cfg.ha_notify_cooldown_seconds
            and epoch - s.notified < cfg.ha_notify_cooldown_seconds):
        log.info("camera %s: arrival within %ss of the last notification; not notifying again",
                 cam.name, cfg.ha_notify_cooldown_seconds)
        s.notify_pending = False
        return 0
    image = _notify_image(cfg, snap, cam.name)
    err = notify_ha(cfg, notify_payload(cam.name, confidence, image=image,
                                        camera_entity=cfg.camera_entities.get(cam.name)))
    if err:
        log.error("camera %s: HA notify failed: %s (will retry next poll)", cam.name, err)
        return 1
    s.notify_pending = False
    s.notified = epoch
    log.info("camera %s: notification sent to %s", cam.name, ",".join(cfg.ha_notify))
    return 0


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
    meta = load_model_meta(cfg.model_path) if model_ok else None
    hello = conn is not None and bool(cfg.ha_notify) and not cfg.ha_dry_run
    for cam in cfg.cameras:
        snap, err = capture_snapshot(cam.url, cfg.snapshot_timeout_s, cfg.snapshot_source)
        if snap is None:
            log.error("camera %s: %s", cam.name, err)
            rc = 1
            continue
        try:
            if hello:  # the first good frame of the poll rides along
                rc |= _notify_hello(cfg, conn, snap, cam.name)
                hello = False
            rc |= _poll_camera(cfg, conn, cam, snap, model_ok, now, dry_run, meta)
        finally:
            # capture_snapshot used a temp file: remove it on every path.
            _remove(snap)
    if hello:  # every camera failed: still say hello, without a photo
        rc |= _notify_hello(cfg, conn)
    if conn is not None:
        conn.close()
    return rc


def save_uncertain(cfg: Config, conn: sqlite3.Connection, camera: str, snap: str, p: float,
                   when: datetime | None = None) -> str | None:
    """Keep the raw frame (no overlay, unlike evidence/) when the model is
    unsure, in capture.py's layout so the training tools read it as is:
    DATA_DIR/frames/<day>/<camera>/<HHMMSS>_uncertain.jpg plus a row in
    DATA_DIR/frames/<day>/index.jsonl. At most one per camera per
    uncertain_every_s; the last save time lives in the db so restarts do
    not reset it. Returns the saved path, or None."""
    if not cfg.uncertain_every_s or not cfg.uncertain_low <= p < cfg.uncertain_high:
        return None
    when = when or datetime.now().astimezone()
    key = f"uncertain_saved_{camera}"
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row and when.timestamp() - float(row[0]) < cfg.uncertain_every_s:
        return None
    day = when.strftime("%Y-%m-%d")
    rel = f"{day}/{camera}/{when.strftime('%H%M%S')}_uncertain.jpg"
    dest = os.path.join(cfg.data_dir, "frames", rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(snap, dest)
    record = {"file": rel, "camera": camera, "ts": when.isoformat(timespec="seconds"),
              "reason": "uncertain", "p": round(p, 3)}
    with open(os.path.join(cfg.data_dir, "frames", day, "index.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(when.timestamp())))
    log.info("camera %s: p_package=%.3f is uncertain; frame kept as %s", camera, p, rel)
    return dest


def classifier_zone(cfg: Config, meta: dict, camera: str):
    """ZONES from the env wins; else the zone the model was trained on."""
    zone = cfg.zones.get(camera) or meta["cameras"].get(camera)
    return tuple(zone) if zone else None


def _poll_camera(cfg: Config, conn: sqlite3.Connection | None, cam: Camera, snap: str,
                 model_ok: bool, now: str, dry_run: bool, meta: dict | None = None) -> int:
    if meta is not None:
        zone = classifier_zone(cfg, meta, cam.name)
        if zone is None:
            if cam.name not in _warned_no_zone:
                _warned_no_zone.add(cam.name)
                log.warning("camera %s: the classifier has no zone for it; skipping", cam.name)
            return 0
        try:
            p = classify(snap, cfg.model_path, zone, meta)
        except Exception:  # noqa: BLE001 — one bad inference must not kill the poll
            log.error("camera %s: inference failed", cam.name, exc_info=True)
            return 1
        detected, best = p >= cfg.confidence, round(p, 3)
        kept = [{"box": zone, "confidence": p}] if detected else []
        log.info("camera %s: p_package=%.3f detected=%s", cam.name, p, detected)
        if conn is not None:
            save_uncertain(cfg, conn, cam.name, snap, p)
    else:
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
    strong = detected and best is not None and best >= cfg.notify_confidence
    s.strong_hits = s.strong_hits + 1 if strong else 0
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
    if cfg.ha_dry_run:
        # ha_synced/ha_posted stay untouched, so switching HA_DRY_RUN off
        # posts the current state on the very next poll.
        if changed:
            log.info("camera %s: state %s -> %s (HA_DRY_RUN: Home Assistant not updated), "
                     "evidence %s", cam.name, old_present, new_present, ev)
            if new_present and cfg.ha_notify:
                log.info("camera %s: HA_DRY_RUN: would notify %s once %d polls reach %.2f",
                         cam.name, ",".join(cfg.ha_notify), cfg.confirm_hits,
                         cfg.notify_confidence)
    else:
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
        rc |= _handle_notify(cfg, cam, s, changed, new_present, best, epoch, snap)
    save_state(conn, cam.name, s)
    conn.commit()
    return rc


def cmd_check(cfg: Config) -> int:
    """Config + ffmpeg + model presence + one snapshot per camera. No writes."""
    problems = []
    print(f"config: {len(cfg.cameras)} camera(s), poll {cfg.poll_interval_seconds}s, "
          f"confidence {cfg.confidence}, notify at {cfg.notify_confidence}, "
          f"confirm {cfg.confirm_hits}, clear {cfg.clear_misses}")
    for cam in cfg.cameras:
        if "_main" in cam.url or "Preview_01_main" in cam.url or "h264Preview_01_main" in cam.url:
            print(f"hint: {cam.name} uses a main-stream URL; use the sub-stream")
    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg not found")
    if not os.path.exists(cfg.model_path):
        problems.append(f"model not found at {cfg.model_path} (capture-only until weights arrive)")
    else:
        try:
            meta = load_model_meta(cfg.model_path)
        except (ConfigError, ValueError) as exc:
            meta = None
            problems.append(str(exc))
        if meta is None:
            print(f"model: box detector {cfg.model_path}")
        else:
            print(f"model: zone classifier {cfg.model_path}, imgsz {meta['imgsz']}, "
                  f"trained on {sorted(meta['cameras'])}")
            for cam in cfg.cameras:
                zone = classifier_zone(cfg, meta, cam.name)
                print(f"  {cam.name}: " + (f"zone {list(zone)}" if zone else "no zone, skipped"))
    if cfg.ha_dry_run:
        print("HA_DRY_RUN=1: state changes are logged, Home Assistant is never called")
    if cfg.ha_notify:
        print(f"notify on arrival: {', '.join(cfg.ha_notify)} "
              f"(cooldown {cfg.ha_notify_cooldown_seconds}s)"
              + (" -- not sent while HA_DRY_RUN=1" if cfg.ha_dry_run else ""))
        print("notify photo: " + (f"on (uploaded to HA /media/{NOTIFY_MEDIA_DIR}/; needs an "
                                  "admin HA_TOKEN)" if cfg.ha_notify_image
                                  else "off (HA_NOTIFY_IMAGE=0)"))
        for cam in cfg.cameras:
            entity = cfg.camera_entities.get(cam.name)
            print(f"notify tap {cam.name}: " + (f"opens {entity}" if entity
                                                else "opens the app (CAMERA_ENTITIES unset)"))
    else:
        print("notify: off (HA_NOTIFY unset; sensor only)")
    for cam in cfg.cameras:
        snap, err = capture_snapshot(cam.url, cfg.snapshot_timeout_s, cfg.snapshot_source)
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
