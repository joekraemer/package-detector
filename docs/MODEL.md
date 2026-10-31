# Model plan

> **Change of plan (2026-10-29): zone classifier, not a detector.**
> The sensor only answers "is there a package in this camera's zone?",
> and the cameras do not move. So v1 crops each camera's zone and runs a
> yes/no image classifier (YOLO11n-cls, ONNX, CPU) on the crop instead
> of a box detector. Labels become "package" / "empty" per frame instead
> of hand-drawn boxes. Labelling is done mostly by a local vision model
> (teacher), checked by a human on the uncertain ones, and the small
> classifier is trained on those labels (student). The sections below
> on detector training are kept for reference until the service switch
> (`detect()` -> `classify()`) lands. Build order:
>
> 1. `capture.py`: collect frames (below, [Collecting frames](#collecting-frames)).
> 2. `autolabel.py`: local VLM teacher + before/after hint -> labels with source and confidence.
> 3. A review page for uncertain and spot-check frames only.
> 4. `train.py`: YOLO11n-cls -> ONNX, scored on a hand-labelled held-out set (split by day).
> 5. Service switch to the classifier.
>
> The held-out evaluation set is always labelled by a human, never by
> the teacher: it is the one honest measure of the model.

## Collecting frames

`capture.py` runs on the fleet MacBook (below), configured by the same
env variables as the detector (see `.env.example`, the `CAPTURE_*`
block). Two cameras: `doorbell` (trigger: the doorbell's
person sensor) and `garage` (trigger: its motion sensor).

```text
python capture.py check                      # config, ffmpeg, HA entities, 1 frame per camera; keeps nothing
python capture.py run                        # baseline + HA-triggered frames, forever
python capture.py session --tag empty        # staged shots: Enter = shoot, 't <tag>' = change tag
```

- **Baseline**: every 20 min by day, every 60 min at night, every 10 min
  within 45 min of sunrise/sunset (IR night-mode switch). About 40-50
  frames per camera per day, so night is represented without flooding
  the labelling queue with empty dark frames.
- **Events**: trigger on -> `arrive` frame; trigger off -> `after_30s`
  and `after_120s` frames, all sharing one `event` id. Deliveries are a
  person visit, so nearly every real positive comes from the `after`
  frames, and the last baseline before `arrive` is the "before" frame.
  Capped at 12 visits per camera per rolling hour.
- **Sessions**: stage packages, tag what is in the zone
  (`brown-box-door`, `white-mailer-mat`, `bag-step`, `empty`), press
  Enter per layout. Session frames label themselves from the tag.

Full frames are kept, never pre-cropped: moving the doorbell or
redrawing a zone means re-cropping, not re-shooting. Layout:
`CAPTURE_DIR/YYYY-MM-DD/<camera>/<HHMMSS>_<reason>.jpg` plus one
`index.jsonl` per day (`file, camera, ts, reason, phase, event?, tag?`).

### Where it runs: the fleet MacBook

Capture runs on the fleet host as its own container, `package-capture`:
the same `ghcr.io/joekraemer/package-detector:main` image as the
detector, with `python capture.py run` as the command. It is behind the
compose profile `package-capture`, so it starts only once
`~/.config/fleet/package-capture.env` exists on the fleet Mac (the
`CAMERAS`, `HA_*` and `CAPTURE_*` lines from `.env.example`;
`CAPTURE_DIR=/data/frames`). To stop capturing, delete the service block
from `fleet/compose.yaml`: deleting the env file also stops it, but then
fleet's health check reports the app as not deployed on every run.
Sub-stream JPEGs are ~100 KB, so expect tens of MB per day for both
cameras.
Frames live in the named volume `package-capture-data`.

On the fleet Mac:

```text
cd ~/fleet
docker compose exec package-capture python capture.py check
docker compose exec -it package-capture python capture.py session --tag empty
docker compose logs --tail 50 package-capture
# copy the frames out (for labelling/training elsewhere):
docker run --rm -v fleet_package-capture-data:/d:ro alpine tar czf - -C /d frames > ~/package-frames.tgz
```

(The volume's full name carries the compose project prefix; check it
with `docker volume ls | grep package-capture`.)

## Target (original detector plan)

YOLO11n, exported to ONNX, CPU inference, one class: `package`.
Input 640 px. The deployed service never sees PyTorch or Ultralytics;
it only runs the `.onnx` file with onnxruntime.

Why YOLO11n: 2.6M parameters, ~56 ms CPU ONNX per image published for
a server-class CPU. At one frame per camera every 15 seconds, inference
speed is not the constraint — accuracy on your porch is. If 11n proves
too weak on small/distant boxes, the step up is 11s (9.4M params,
~90 ms), not a bigger jump; the service code does not care which size
the ONNX file is, so swapping is a config + file change, not a code
change.

## Why there is no model yet

A stock COCO YOLO model has no package class. Any "download a package
model" shortcut needs vetting: whose porch data, what licence, what
false-positive rate on empty porches, and whether the weights are even
still available. That vetting is issue #1, and it is a real decision,
not a download link to paste in. The expected outcome is one of:

1. A public fine-tune that survives evaluation on your own porch
   images (unlikely to be perfect, but a usable starting point), or
2. Fine-tuning YOLO11n yourself from COCO weights on your own labelled
   porch/garage images plus a public box dataset.

Either way, the model that ships in v1 must be evaluated on held-out
images from *your* cameras, not on the training set and not on stock
photos of porches.

## Dataset, in priority order

1. **Your own cameras (highest value).** Save snapshots at all times
   of day, all weather, porch empty and with packages of every size
   you actually receive. Aim for 100–200 labelled images to start:
   positives (box, mailer, bag, multiple packages, partially occluded
   by the door frame / railing) and, more importantly, hard negatives
   (see below). The evidence snapshots this service saves on state
   changes are a free, ongoing source of exactly these images.
2. **Public box datasets (base only).** Evaluate in issue #1 for
   licence, label quality, and how much of it is warehouse/conveyor
   imagery (which teaches "box" but not "porch"). Use it to pre-train,
   then fine-tune on (1).
3. **Synthetic augmentation, sparingly.** Flips and brightness/contrast
   shifts are safe. Pasting clip-art boxes onto porch photos is not:
   it teaches the model that boxes have clean cut-out edges.

Hard negatives to collect deliberately — these are where porch
detectors actually fail, per published porch-detection write-ups:

- empty porch at dawn, noon, dusk, night/IR
- doormat alone, shoes, umbrella stand, plant pots
- a person standing on the porch, a visitor carrying a box (transient)
- seasonal clutter: decorations, pumpkins, snow piles, fallen leaves
- wet porch reflections, strong shadows, glare

## Labelling

- One class, `package`. Box the whole package, not the label on it.
- A padded mailer lying flat counts; a letter in the mailbox does not
  (it is not on the porch floor — zone it out if the mailbox is in
  frame, or exclude it in labelling and let the zone do the work).
- If two packages touch, box them separately where a human can see the
  seam; if they read as one lump, one box is fine — the sensor only
  answers yes/no.
- Keep a `notes.md` next to the dataset: camera, date range, what
  changed on the porch (new doormat? new wedge angle on the doorbell?
  — the doorbell wedge changes the view, so re-check zones after any
  physical change).

## Training sketch (offline, not in the deployed image)

```text
# On a machine with a GPU, or CPU if patient (11n is small):
# 1. Start from yolov8/11n COCO weights (ultralytics).
# 2. Train on public box data, then fine-tune on your labelled set
#    (or train on the combined set with your images up-weighted).
# 3. Export to ONNX (opset compatible with the pinned onnxruntime).
# 4. Evaluate on a held-out set from your cameras, never trained on.
# 5. Copy the .onnx to the fleet host's models mount; set MODEL_PATH.
```

Record, with the weights, wherever they end up living: base model,
training data (counts, sources, licences), training command/epochs,
and the held-out evaluation numbers. A model file with no provenance
is un-debuggable the first time it false-alarms.

## Evaluation gate for v1

On a held-out set of your own porch/garage images (at least 50 empty,
at least 30 with packages, across day/night):

- false positives on empty-porch images: near zero. A sensor that
  cries package on an empty porch will be ignored, which is worse
  than no sensor.
- misses on real packages: report honestly, then decide whether the
  threshold, the zone, or more training data is the fix. Do not just
  lower the threshold until the misses go away — that is how you buy
  false positives back.

After deployment, the tuning loop is: `--dry-run` + evidence review
-> label the mistakes -> retrain -> re-evaluate on the held-out set
-> redeploy the weights (no code change needed).

## Open questions (see issues)

- Envelopes: flat mailers vs doormat is the hardest class boundary.
  v1 may ship box-only and treat flat mailers as a follow-up, or ship
  one class and accept envelope misses. Decide from the first labelled
  batch, not in advance. (issue #3)
- Night/IR: separate threshold, or daylight-only mode? (issue #4)
- Garage camera: inside looking at the floor, or outside looking at
  the driveway? The zone and the negatives differ completely.
  (issue #2 — needs Joe's camera list)
