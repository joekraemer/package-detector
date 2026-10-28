# Model plan

## Target

YOLO11n, exported to ONNX, CPU inference, one class: `package`.
Input 640 px. The deployed service never sees PyTorch or Ultralytics;
it only runs the `.onnx` file with onnxruntime.

Why YOLO11n: 2.6M parameters, ~56 ms CPU ONNX per image published for
a server-class CPU. At one frame per camera every 3 minutes, inference
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
