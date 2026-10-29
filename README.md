# package-detector

Polls your Reolink cameras every 15 seconds, runs a lightweight YOLO
model looking for a package, and flips a Home Assistant sensor per camera:
`binary_sensor.package_<camera>` — on when a package is present, off when
it is gone.

The reminder that actually gets you to grab it is a Home Assistant
automation (see [Home Assistant](#home-assistant)): when you arrive home
and a package sensor is on, you get a notification. A push the moment a
package lands, while you are out, is exactly the kind you forget.

This repo is deliberately modelled on
[birdlisten](https://github.com/joekraemer/birdlisten): same camera
connection method, same env-file config, same container shape, same
fleet deploy (push to `main` is the deploy). Where it differs, it says so,
and why.

```text
camera 1 RTSP sub-stream ─┐
camera 2 RTSP sub-stream ─┼─▶ ffmpeg snapshot (1 frame, every 15 s) ─▶ YOLO11n ONNX ─▶ state machine ─▶ HA sensor
   ...                    ┘                                                              └─▶ SQLite + evidence snapshot
```

## Status: scaffold, not yet a working detector

What is in here and verified: config parsing and validation, the
per-camera present/absent state machine, zone filtering, the Home
Assistant state payload, snapshot capture via ffmpeg, and the test suite
for all of that — all without a camera or a model, by design.

What is **not** in here yet, and is the real work:

- **No model weights are shipped.** A stock YOLO model has no "package"
  class — it knows person, car, dog, chair, but not box-on-porch. This
  needs a YOLO11n model fine-tuned on packages, which does not exist in
  this repo yet. See [Model](#model) and issue #1.
- **Not verified against a real camera.** Snapshot capture, the NVR
  session behaviour, and the zone geometry for your porch/garage views
  are all untested. See issue #2.
- **No Home Assistant token or camera password is in this repo**, ever.
  They live only in `.env` / the fleet env file (see Setup).

## Why polling, not birdlisten's always-on stream

birdlisten keeps one ffmpeg per camera open permanently because audio
must never be missed. Packages do not need that: a box sits there for
hours. Polling one frame every 15 seconds is:

- still light on the host (a fraction of a second of model inference
  per frame, idle the rest — even at 15 s cadence),
- a short RTSP session per poll, closed again, instead of a permanent
  session per camera (birdlisten documents NVR refusals at
  `453 Not Enough Bandwidth` when too many permanent sessions stack
  up) — but note the new trade-off: at 15 s, that is 4 session
  open/closes per camera per minute, so NVR session churn is the thing
  to watch in real-camera testing (issue #2),
- and a better match for the job: the output is a slow-changing state
  ("package on porch: yes/no"), not a stream of events.

The trade-off is latency: a package can sit undetected for up to one
poll interval plus the time the poll itself takes (snapshots are taken
in turn, and loop.py sleeps the interval *after* a poll finishes, so
the real period is poll duration + 15 s). At 15 s that is responsive
enough to feel live, and still useless as a doorbell. This is not a
doorbell.

## Camera connection (same method as birdlisten)

- Reolink RTSP sub-stream URL, per camera:
  `rtsp://USER:PASS@IP:554/h264Preview_01_sub` (older firmware) or
  `rtsp://USER:PASS@IP:554/Preview_01_sub` (newer).
- Use the **sub-stream**: same scene at a fraction of the bandwidth, and
  plenty of resolution for a box on a porch. A main-stream URL gets a
  startup warning, exactly as in birdlisten.
- Give the cameras fixed IPs (DHCP reservation in the eero app).
- Test from any laptop first: `ffplay -rtsp_transport tcp <url>`.
- The doorbell in this house is a black Reolink D340W: it has **no
  native package detection** (Reolink restricts that to the white 3:4
  model), which is why this project exists at all. Native detection is
  not a fallback for the doorbell; for any other camera, check its own
  app first before adding it here.

One snapshot per camera per poll is taken with ffmpeg, the same tool
birdlisten already uses:

```text
ffmpeg -rtsp_transport tcp -i <rtsp-url> -frames:v 1 -q:v 2 snapshot.jpg
```

with a 15 s timeout. A camera that fails to give a frame is logged and
skipped for that poll; it does not flip any state and does not kill the
run. Three consecutive failed polls for one camera are a WARNING worth
looking at, not a crash.

## Setup

1. `cp .env.example .env`, fill in `CAMERAS`, `HA_URL`, `HA_TOKEN`.
   Never commit `.env`.
2. Prove it, in this order:
   ```text
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--check" app
       # config, ffmpeg, model file presence, 1 snapshot per camera, no writes, no HA calls
   docker compose run --rm -e LOOP_ONCE=1 -e RUN_ARGS="--dry-run" app
       # one real poll: snapshot + inference, prints what it would set in HA, writes nothing
   docker compose run --rm -e LOOP_ONCE=1 app
       # one real poll, writes state + HA
   docker compose up
       # the real loop
   ```
3. Fleet: on the host, create `~/.config/fleet/package-detector.env`
   with the same content as `.env`. Then in `fleet/compose.yaml`
   add a `package-detector` service (no block exists there yet — a
   2026-10-29 read-only audit of the fleet repo confirmed this; same
   shape as `birdlisten`: image
   `ghcr.io/joekraemer/package-detector:main`, `env_file`, a volume
   at `/data`, `restart: unless-stopped`), push fleet. The build
   workflow in [Deploy](#deploy) is in place (added 2026-10-29,
   issue #7 closed), so pushes to this repo's `main` build and
   publish the image to GHCR; the fleet host pulling
   that tag is what deploys it. The fleet repo itself is private, so
   its self-hosted health runner is not exposed to public pull
   requests.

## Verifying cameras (issue #2)

This step runs on the LAN, next to the cameras — not from a remote
agent, which cannot reach them. `verification-capture.sh` reads
`CAMERAS` from `.env`, grabs one ffmpeg snapshot per camera, and saves
them timestamped under `verification/`, with a log that records only
the camera host, never the user or password.

```text
bash verification-capture.sh          # one round, now
bash verification-capture.sh 40 15    # churn test: 40 rounds, 15 s apart (~10 min)
```

Check `verification/capture.log` for failures. A `453` refusal or a
run of timeouts during the churn test means the NVR is unhappy with
15-second session churn (see issue #2) — that verdict, plus snapshots
at day, dusk, and night/IR, empty and with a staged box/mailer where
deliveries actually land, are what set the per-camera zones. Note:
`--check` reports "model not found" and exits 1 until issue #1 lands;
for camera verification, the snapshot OK lines are the signal, not
the exit code.

## Config

| var | default | meaning |
|---|---|---|
| `CAMERAS` | — (required) | `name=rtsp-url` pairs, comma separated, same format as birdlisten |
| `POLL_INTERVAL_SECONDS` | `15` | seconds between polls, 15..900. 15 is the default; raise it if the host or NVR struggles |
| `MODEL_PATH` | `/models/package.onnx` | YOLO11n ONNX model, single `package` class. Mounted, not committed (see Model) |
| `CONFIDENCE` | `0.5` | detection confidence bar, 0..1. Start here, tune from evidence (see Tuning) |
| `CONFIRM_HITS` | `2` | consecutive polls with a detection before a camera flips to present (~30 s at 15 s polls) |
| `CLEAR_MISSES` | `5` | consecutive polls without one before a camera flips to absent (~75 s at 15 s polls — a single dark/occluded frame must not clear a real package) |
| `ZONES` | (unset = full frame) | per-camera rectangle, `porch=0,0.5,1,1` = x_min,y_min,x_max,y_max as fractions of the frame. Only detections whose centre falls inside count |
| `HA_URL` | — (required for live runs) | Home Assistant base URL, e.g. `http://homeassistant.local:8123` |
| `HA_TOKEN` | — (required for live runs) | HA long-lived access token. Secret: fleet env file only, never committed |
| `DATA_DIR` | `/data` | SQLite db + evidence snapshots |
| `SNAPSHOT_TIMEOUT_S` | `15` | ffmpeg snapshot timeout per camera |
| `TZ` | `America/Los_Angeles` | timestamps |

A bad value is a config error (`config error: ...`, exit code 2), same
convention as birdlisten. `--check` and `--dry-run` do not need
`HA_URL`/`HA_TOKEN`; a live run does.

## How a poll works

1. For each camera, in turn: grab one frame with ffmpeg.
2. Run the ONNX model on that frame (640 px input, CPU only).
3. Drop detections below `CONFIDENCE`, and drop any whose centre is
   outside that camera's zone.
4. Feed the result into the per-camera state machine:
   - present requires `CONFIRM_HITS` consecutive detections — one stray
     frame (a visitor carrying a box, a shadow, a doormat that looks
     boxy for a second) must not flip the sensor;
   - absent requires `CLEAR_MISSES` consecutive misses — one dark or
     blurred frame must not clear a real package.
5. Only on a state **change** does it write to HA and save an annotated
   evidence snapshot under `/data/evidence/YYYY-MM-DD/<camera>_<time>.jpg`.
   Steady state writes nothing to HA beyond the poll log row, so HA
   history is not spammed every poll.
6. Every poll appends one row per camera to SQLite (`polls`), so tuning
   later has real data: timestamp, camera, detection yes/no, best
   confidence, state after the poll.

`--report N` prints, per camera, the current state, since when, and the
last 7 days of transitions — the thing to read before moving thresholds.

## Home Assistant

The service sets, per camera:

```text
POST {HA_URL}/api/states/binary_sensor.package_<camera>
{"state": "on"|"off",
 "attributes": {"friendly_name": "Package <camera>",
                "confidence": 0.83, "last_seen": "<iso>",
                "device_class": "occupancy"}}
```

Suggested automation (in HA, not in this repo — HA owns automations):

```yaml
# Notify on arrival home, only if a package is actually waiting.
- alias: "Package waiting when arriving home"
  trigger:
    - platform: zone
      entity_id: person.joe
      zone: zone.home
      event: enter
  condition:
    - condition: state
      entity_id: binary_sensor.package_porch
      state: "on"
  action:
    - service: notify.mobile_app_joe
      data:
        title: "Package on porch"
        message: "There's a package on the porch — grab it on your way in."
```

Add one automation per camera, or a template sensor that ORs them.
If you want a push the moment a package is confirmed, not just on
arrival, that is a second automation on the sensor turning on — do not
make this service send pushes itself. One job, one output: the sensor.
ntfy (birdlisten's push path) is deliberately not wired in here; see
issue #6 if you want it as an option.

## Model

Design target: **YOLO11n, ONNX, CPU-only, single class `package`**.
YOLO11n is 2.6M parameters; published CPU ONNX speed is ~56 ms per
640 px image on a server-class CPU (your host will be slower, but at
one frame per camera every 15 seconds, even 10x slower is nothing).
The runtime image therefore carries **onnxruntime only — no PyTorch,
no Ultralytics**. Training (Ultralytics + PyTorch) is a separate,
offline tool, never in the deployed image; that is what keeps the
deployed image light.

Classes, v1: `package` = a cardboard box, padded mailer, or shipping
bag sitting on the porch/garage floor. It is one class on purpose:
splitting box vs envelope vs bag on day one triples the labelling for
no benefit to the sensor, which only answers yes/no. Envelopes are a
known hard case (flat, low contrast against a doormat) — see issue #3.

Training data plan (full detail in `docs/MODEL.md`):

1. **Your own porch/garage first.** The evidence snapshots this service
   saves, plus a week of saved snapshots at all times of day, are the
   highest-value data there is: same camera angle, same light, same
   doormat. Label 100–200 of those before reaching for anything public.
2. **A public box dataset second**, as a base to fine-tune from — to be
   evaluated for licence and quality in issue #1, not assumed. Public
   porch datasets skew to other people's porches; they help the model
   learn "box", they do not teach it your porch.
3. **Hard negatives matter more than positives**: empty porch at dawn,
   noon, dusk, night/IR; doormat alone; shoes; a visitor standing there;
   Christmas decorations; rain and snow. Most false positives in porch
   detection stories are seasonal clutter, not boxes.

Evaluation before it is trusted: a held-out set of your own porch
images, labelled, with a false-positive rate target of near zero on
empty-porch images — a sensor that cries package every other day will
be ignored, which is worse than no sensor. The `--dry-run` + evidence
review loop is how you get there: run, look at the evidence, label the
mistakes, retrain, repeat.

## Deploy

`.github/workflows/build.yml` is intended to mirror birdlisten's: on
push to `main` it builds `linux/amd64` and publishes
`ghcr.io/joekraemer/package-detector:main` plus a `:sha-<short>` tag
for pinning back a bad release, then smoke-tests the image
(`--help` must work). "Push to main IS the deploy" — the fleet host
pulls `:main` on its own schedule.

That workflow file landed on `main` 2026-10-29 (issue #7 — it could
not be pushed via the GitHub integration, which needs a separate
workflow permission, so it was added via the GitHub web UI) and the
first GHCR image has published, so a fleet service block now has
something to pull.

Divergences from birdlisten, on purpose:

- **Polling, not streaming** (see above).
- **uv + `uv.lock`, not pip + `requirements.txt`.** The scaffold
  started on pip because uv was not available where it was written;
  this was migrated to match the fleet app contract (issue #5) and
  birdlisten exactly: `pyproject.toml` with `[tool.uv] package = false`,
  Python 3.12 (not birdlisten's tflite-pinned 3.11), uv 0.11 in the
  Dockerfile, `uv sync --frozen --no-dev --no-install-project`.
- **No web UI in v1.** birdlisten's collage page earns its keep because
  there is something to look at. Here the HA sensor is the UI; the
  evidence snapshots are files, not a page. A status page is part of
  issue #6 (optional outputs).
- **amd64 only**, same as birdlisten (the fleet host is Intel). If the
  home-lab host is ever ARM, add `linux/arm64` to the workflow — no
  issue filed for that yet.

## Tuning

- Start: `CONFIDENCE=0.5`, `CONFIRM_HITS=2`, `CLEAR_MISSES=5`, `POLL_INTERVAL_SECONDS=15`.
- If it misses real packages: lower confidence first, then check the
  zone — a box outside the zone is invisible by design, and the
  evidence snapshot will show whether the box was detected but zoned
  out, or not detected at all. Those are different fixes.
- If it false-alarms: raise confidence, then add the offending scene
  to the hard-negative set and retrain. Raising the threshold alone
  just trades false positives for misses.
- Night/IR: expect worse accuracy. If night false positives dominate,
  consider a higher night confidence or a "daylight only" mode as a
  config option, rather than tuning the daytime threshold to fix the
  night — that is issue #4.

## Files

| file | what it is |
|---|---|
| `packagedetector.py` | the whole service: config, capture, inference, state, storage, HA |
| `loop.py` | container PID 1: runs one poll, sleeps `LOOP_INTERVAL`, repeats; `LOOP_ONCE=1` runs once (birdlisten pattern) |
| `Dockerfile`, `compose.yaml`, `.env.example` | dev-laptop run, same shape as birdlisten |
| `pyproject.toml`, `uv.lock` | uv environment (fleet app contract, issue #5); `uv.lock` is committed and frozen in the image build |
| `docs/MODEL.md` | model + dataset + evaluation plan |
| `tests/` | stdlib unittest: config, zones, state machine, HA payload — no camera, no model, no network |
| `verification-capture.sh` | issue #2 capture tool: one snapshot per camera per round, host-only log, no credentials logged |
