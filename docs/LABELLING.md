# Labelling guide (issue #4)

How to turn the service's evidence snapshots into a labelled dataset
that can actually train v1 — and how to keep that dataset honest over
time. This guide covers the workflow only. No labelled images ship in
this repo: porch images stay on the home network, and the dataset
itself is never committed.

Read this alongside `docs/MODEL.md`, which owns the model plan, the
training sketch, and the v1 evaluation gate. Where the two disagree,
`MODEL.md` wins and this guide should be fixed.

## Where the data lives

- **Raw evidence:** the service saves an annotated snapshot on every
  state change under `/data/evidence/YYYY-MM-DD/` on the fleet host
  (see `DATA_DIR`). That is the free, ongoing source of training
  images — positives and, more importantly, mistakes.
- **Labelled set:** keep it on the labelling machine, outside the repo
  checkout (for example `~/package-detector-dataset/`), in per-camera
  folders: `dataset/<camera>/images/`, `dataset/<camera>/labels/`,
  plus one `dataset/notes.md` (see below). Never commit it, and never
  upload porch images to a hosted labelling service — local tools
  only, per issue #4.
- **Held-out set:** a separate folder per camera,
  `dataset/<camera>/heldout/`, locked the day it is set aside. It is
  never trained on, never labelled-and-then-folded-back-in, and never
  deleted. If the porch view changes physically (see below), start a
  second held-out folder dated from the change; do not silently mix
  pre- and post-change images.

## Tool and format

Use a local, offline labeller that writes YOLO format natively
(labelImg is the boring default: one class file, one `.txt` per
image). Format per image:

```text
# classes.txt — the whole file:
package
# <image>.txt — one line per box, class 0, normalised coordinates:
0 <x_center> <y_center> <width> <height>
```

An image with no boxes gets an empty `.txt` file. That is how a hard
negative is recorded — an empty label file is data, not a missing
label.

## Step 1: hold out first, before anything is trained on

Before labelling for training, set aside the held-out set per camera
from `MODEL.md`'s gate: at least **50 empty + 30 with-package**
images, across day, dusk, and night/IR. Pick them first, move them to
`heldout/`, and do not touch them again except to evaluate. A held-out
set chosen after training, or cherry-picked to flatter the model, is
worthless — this is the one step that cannot be redone honestly.

## Step 2: label positives — the box rules

These restate `MODEL.md`'s rules, plus the two porch-specific calls:

- **One class, `package`.** Box the whole package, not the label on
  it. Box the visible portion only when it is partly occluded by the
  door frame or railing.
- **A padded mailer lying flat counts.** A letter in the mailbox
  does not — it is not on the porch floor. Zone the mailbox out if
  it is in frame, or leave it unboxed and let the zone do the work.
- **Touching packages:** box them separately where a human can see
  the seam; if they read as one lump, one box is fine — the sensor
  only answers yes/no.
- **Doorbell mirror strip:** the convex mirror shows a real view of
  the doorstep, higher in the frame. Box a package visible in the
  mirror strip as its own box, just like a direct-view package.
- **Reflections are not packages:** a package reflected in a wet
  porch floor or a window is a hard negative, not a second box. The
  mirror strip above is the exception because it is the camera's
  actual view, not a surface reflection.
- **A visitor carrying a box is a hard negative.** The box is
  transient — it will be gone in seconds, and labelling it teaches
  the sensor to flip on every visitor. Only box a package once it
  has been set down where deliveries land.

## Step 3: collect hard negatives deliberately

Empty-porch images alone are not enough. Per `MODEL.md`, collect
these on purpose, across day/dusk/night-IR, and label them with empty
`.txt` files:

- empty porch at dawn, noon, dusk, night/IR
- doormat alone, shoes, umbrella stand, plant pots
- a person standing on the porch, a visitor carrying a box
- seasonal clutter: decorations, pumpkins, snow piles, fallen leaves
- wet porch reflections, strong shadows, glare
- the garage view's own equivalents, once the garage camera's view
  is known (issue #2) — its negatives differ completely from the
  porch's

## Step 4: review pass and counts

Every labelled batch gets a second look before it is used: seam
calls, mirror-strip vs reflection calls, and mailer-vs-doormat calls
are where two labellers (or one labeller on two days) disagree, so
those are the ones to re-check. Then record the batch in
`dataset/notes.md`:

```text
## 2026-11-xx — porch batch 1
- camera: porch (doorbell), date range: ...
- added: N positives, N hard negatives; held-out: 50 empty / 30 with-package (locked, untouched)
- physical changes since last batch: none / new doormat on <date> / mirror moved on <date>
- disagreements resolved: ...
```

Counts live in `notes.md` and get reported back to issue #4 when a
batch lands. A dataset with no counts is un-debuggable.

## Retention rule (issue #4 acceptance, manual step)

No retention code ships with this guide — this is the documented
manual rule, with a date, which issue #4 accepts:

- **State-change evidence snapshots** (`/data/evidence/`): keep
  **90 days** on the fleet volume, then prune anything older that
  has not been copied into the labelled set. Prune monthly; note
  the prune date in `dataset/notes.md`.
- **Curated labelled set** (images + `.txt` + `notes.md`): keep
  **indefinitely**, with a second copy somewhere else on the home
  network. This is the irreplaceable part — re-labelling 100–200
  images is the real cost, not disk space.
- **Held-out set:** never pruned, never folded into training.

## Retrain trigger and weight swap (issue #4)

- **Trigger:** after **20 newly labelled mistakes** (false positives
  or misses found in evidence review), or **monthly**, whichever
  comes first.
- **Procedure:** retrain offline per `MODEL.md`'s training sketch,
  then re-evaluate on the held-out set **before** swapping weights.
  Report frame level and sensor level separately (the 2-hit/5-miss
  state machine hides frame noise). Swap `MODEL_PATH` only if
  held-out false positives are near zero and misses are no worse
  than the current weights. Never swap on training-set numbers.
- **Provenance:** record with the weights wherever they end up
  living — base model, training data (counts, sources, licences),
  training command/epochs, and the held-out evaluation numbers. A
  model file with no provenance is un-debuggable the first time it
  false-alarms.
- **View changes:** note any physical change (new doormat, mirror
  moved, doorbell mount changed) in `dataset/notes.md` with its date,
  and re-check zones after it — a view change silently invalidates
  old zones and old held-out images.

## What this guide does not do

- It ships no labelled images and runs no retraining — there are no
  porch images to label yet (issue #2) and no model to retrain (issue
  #1). Issue #4 therefore stays open: its acceptance criteria (a
  labelled set with recorded counts, and at least one documented
  retrain run) are still to be met.
- It does not settle the flat-mailer question for v1 (box-only vs
  mailers included) — `MODEL.md` says decide from the first labelled
  batch, not in advance (issue #3).
