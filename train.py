#!/usr/bin/env python3
"""Train the zone classifier: is there a package in this camera's zone?

Runs on the laptop, in its own environment (PyTorch is never part of the
service image):

  uv venv --python 3.12 .venv-train
  VIRTUAL_ENV=.venv-train uv pip install -r requirements-train.txt
  .venv-train/bin/python train.py prepare      # build the dataset
  .venv-train/bin/python train.py fit          # train, evaluate, export ONNX

Labels, highest priority first:
  1. package-data/frames/labels.jsonl (review page; "unsure" = left out)
  2. session frames: tag "empty" -> empty, any other tag -> package
  3. any other frame -> empty, EXCEPT on a day with sessions on that
     camera (staged packages sit in baselines and visit frames too, so
     those must be labelled explicitly)

Split, so near-identical frames never land on both sides:
  * negatives: whole days. The newest day with no sessions is validation.
  * positives: time blocks. The newest VAL_POS_SHARE of positive frames,
    by time, are validation.

Copy-paste augmentation (train split only): package cutouts from
package-data/boxes.json ({file: [x0, y0, x1, y1]} as frame fractions)
are pasted onto empty training crops at random spots, sizes and light,
and the same number of empty crops get a pasted patch of their own floor
so the paste seam itself is not a clue.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import review

REPO = Path(__file__).resolve().parent
DATA = REPO / "package-data" / "frames"
OUT = REPO / "package-data" / "dataset"
BOXES = REPO / "package-data" / "boxes.json"
MODELS = REPO / "models"
CROP_PX = 256  # saved crop size; training resizes to IMGSZ
IMGSZ = 224
VAL_POS_SHARE = 0.3
NEG_PER_POS = 4  # cap negatives at this many per (real + pasted) positive
PASTES = 300


# ------------------------------------------------------------ labelling

def resolve_labels(rows: list[dict], human: dict[str, str]) -> dict[str, str]:
    """file -> 'package' | 'empty' for every frame we are willing to train on."""
    session_days = {(r["camera"], r["file"][:10]) for r in rows if r["reason"] == "session"}
    out: dict[str, str] = {}
    for r in rows:
        f = r["file"]
        if f in human:
            if human[f] in ("package", "empty"):
                out[f] = human[f]
            continue  # unsure: leave out
        if r["reason"] == "session":
            out[f] = "empty" if r.get("tag") == "empty" else "package"
        elif (r["camera"], f[:10]) not in session_days:
            out[f] = "empty"
    return out


def split(rows: list[dict], labels: dict[str, str]) -> dict[str, str]:
    """file -> 'train' | 'val' (time-ordered blocks, never shuffled)."""
    by_file = {r["file"]: r for r in rows}
    session_days = {r["file"][:10] for r in rows if r["reason"] == "session"}
    neg_days = sorted({f[:10] for f, l in labels.items() if l == "empty"} - session_days)
    val_day = neg_days[-1] if len(neg_days) > 1 else None
    pos = sorted((f for f, l in labels.items() if l == "package"), key=lambda f: by_file[f]["ts"])
    # Real deliveries on the validation day stay with that day; of the rest,
    # the newest VAL_POS_SHARE go to val.
    val_pos = {f for f in pos if f[:10] == val_day}
    rest = [f for f in pos if f not in val_pos]
    n_val = max(1, round(len(rest) * VAL_POS_SHARE)) if len(rest) > 1 else 0
    val_pos |= set(rest[len(rest) - n_val:])
    out = {}
    for f, l in labels.items():
        if l == "package":
            out[f] = "val" if f in val_pos else "train"
        else:
            out[f] = "val" if f[:10] == val_day else "train"
    return out


# ------------------------------------------------------------ images

def zone_crop(im: Image.Image, zone: list[float]) -> Image.Image:
    w, h = im.size
    return im.crop((int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)))


def to_square(im: Image.Image, size: int = CROP_PX) -> Image.Image:
    """Stretch the zone crop to a square. The service must do the same.

    Not letterboxed (bars waste half the pixels of a wide zone) and not
    centre-cropped (that would cut the zone's edges). A fixed stretch is
    harmless: train and inference see the same geometry.
    """
    return im.convert("RGB").resize((size, size), Image.BILINEAR)


def is_night(im: Image.Image) -> bool:
    """IR frames are effectively grey: almost no saturation."""
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean()) < 12


def soft_mask(w: int, h: int, feather: float = 0.18) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    inset_x, inset_y = int(w * feather / 2), int(h * feather / 2)
    m.paste(255, (inset_x, inset_y, w - inset_x, h - inset_y))
    return m.filter(ImageFilter.GaussianBlur(max(1, min(w, h) * feather / 2)))


def cutout(frame: Path, box: list[float], margin: float = 0.15) -> Image.Image:
    im = Image.open(frame).convert("RGB")
    w, h = im.size
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    return im.crop((int(box[0] * w - bw * margin), int(box[1] * h - bh * margin),
                    int(box[2] * w + bw * margin), int(box[3] * h + bh * margin)))


FLOOR_TOP = 0.45  # doorbell zone: the floor starts ~45% down the crop


def paste(base: Image.Image, patch: Image.Image, rng: random.Random, floor_top: float = FLOOR_TOP) -> Image.Image:
    """Blend patch onto base at a random floor spot, matched to local light."""
    base = base.copy()
    s = rng.uniform(0.7, 1.4)
    pw, ph = max(4, int(patch.width * s)), max(4, int(patch.height * s))
    if pw >= base.width or ph >= base.height:
        return base
    patch = patch.resize((pw, ph))
    if rng.random() < 0.5:
        patch = ImageOps.mirror(patch)
    x = rng.randint(0, base.width - pw)
    y = rng.randint(int(base.height * floor_top), base.height - ph)
    region = base.crop((x, y, x + pw, y + ph))
    if is_night(base):
        patch = ImageOps.grayscale(patch).convert("RGB")
    # Match brightness to the spot it lands on, with a little jitter.
    gain = (np.asarray(region, np.float32).mean() + 1) / (np.asarray(patch, np.float32).mean() + 1)
    gain *= rng.uniform(0.85, 1.15)
    arr = np.clip(np.asarray(patch, np.float32) * gain, 0, 255).astype(np.uint8)
    base.paste(Image.fromarray(arr), (x, y), soft_mask(pw, ph))
    return base


# ------------------------------------------------------------ prepare

def cmd_prepare(data: Path, out: Path, camera: str, seed: int) -> int:
    rng = random.Random(seed)
    zones = review.load_zones()
    if camera not in zones:
        print(f"no zone for {camera}: draw one on the review page (/zones) first", file=sys.stderr)
        return 1
    zone = zones[camera]
    rows = [r for r in review.load_index(data) if r["camera"] == camera]
    human = review.read_labels(data / "labels.jsonl")
    labels = resolve_labels(rows, human)
    sides = split(rows, labels)

    if out.exists():
        shutil.rmtree(out)
    for side in ("train", "val"):
        for cls in ("package", "empty"):
            (out / side / cls).mkdir(parents=True)

    def crop_of(f: str) -> Image.Image:
        return zone_crop(Image.open(data / f).convert("RGB"), zone)

    def save(im: Image.Image, side: str, cls: str, name: str):
        to_square(im).save(out / side / cls / f"{name}.jpg", quality=92)

    pos = {s: [f for f, l in labels.items() if l == "package" and sides[f] == s] for s in ("train", "val")}
    neg = {s: [f for f, l in labels.items() if l == "empty" and sides[f] == s] for s in ("train", "val")}
    for s in ("train", "val"):
        rng.shuffle(neg[s])

    # Cutouts only from TRAIN positives, so nothing in val leaks into train.
    boxes = json.loads(BOXES.read_text()) if BOXES.is_file() else {}
    cutouts = [cutout(data / f, b) for f, b in boxes.items() if sides.get(f) == "train"]
    n_paste = PASTES if cutouts else 0

    counts = Counter()
    for s in ("train", "val"):
        for f in pos[s]:
            save(crop_of(f), s, "package", f.replace("/", "_")[:-4])
            counts[s, "package"] += 1
    # Negatives: cap the ratio so the model can't win by always saying empty.
    # Val keeps every negative of its day: the false-positive rate on real
    # empty frames is the number that matters most.
    cap = {"train": NEG_PER_POS * (len(pos["train"]) + n_paste), "val": len(neg["val"])}
    for s in ("train", "val"):
        for f in neg[s][:cap[s]]:
            save(crop_of(f), s, "empty", f.replace("/", "_")[:-4])
            counts[s, "empty"] += 1

    # Copy-paste positives onto train negatives, plus matched floor-patch negatives.
    bases = neg["train"] or []
    for i in range(n_paste):
        base = crop_of(rng.choice(bases))
        save(paste(base, rng.choice(cutouts), rng), "train", "package", f"paste_{i:04d}")
        base2 = crop_of(rng.choice(bases))
        w, h = base2.size
        pw, ph = rng.randint(w // 12, w // 5), rng.randint(h // 10, h // 4)
        px, py = rng.randint(0, w - pw), rng.randint(int(h * FLOOR_TOP), h - ph)
        floor = base2.crop((px, py, px + pw, py + ph))
        save(paste(base2, floor, rng), "train", "empty", f"floorpaste_{i:04d}")
        counts["train", "package"] += 1
        counts["train", "empty"] += 1

    manifest = {"camera": camera, "zone": zone, "seed": seed, "counts": {f"{s}/{c}": n for (s, c), n in counts.items()},
                "val_positive_files": sorted(pos["val"]), "cutouts_from": sorted(f for f in boxes if sides.get(f) == "train")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


# ------------------------------------------------------------ fit

def cmd_fit(out: Path, epochs: int, seed: int) -> int:
    import os
    # Ultralytics pip-installs missing extras on its own; keep the pinned env.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO  # training env only

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolo11n-cls.pt")
    model.train(data=str(out), epochs=epochs, imgsz=IMGSZ, batch=32, device=device, seed=seed,
                project=str(REPO / "package-data" / "runs"), name="cls", exist_ok=True,
                fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.5, erasing=0.3,
                workers=0, plots=False, verbose=False)
    best = REPO / "package-data" / "runs" / "cls" / "weights" / "best.pt"
    model = YOLO(str(best))

    # Evaluate on val, file by file, so mistakes can be looked at.
    report = evaluate(model, out / "val")
    (out / "val_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
    for e in report["errors"][:30]:
        print("  wrong:", e)

    return cmd_export()


def cmd_export() -> int:
    """best.pt -> models/package-cls.onnx, checked against PyTorch on val."""
    import os
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import onnxruntime as ort
    from ultralytics import YOLO

    best = REPO / "package-data" / "runs" / "cls" / "weights" / "best.pt"
    model = YOLO(str(best))
    onnx_path = Path(model.export(format="onnx", imgsz=IMGSZ, opset=17, dynamic=False))
    MODELS.mkdir(exist_ok=True)
    dest = MODELS / "package-cls.onnx"
    shutil.copy(onnx_path, dest)
    (MODELS / "package-cls.json").write_text(json.dumps(
        {"classes": [model.names[i] for i in sorted(model.names)], "imgsz": IMGSZ,
         "zone": review.load_zones(), "preprocess": "zone crop -> stretch to imgsz square -> RGB/255, NCHW"},
        indent=2) + "\n")

    # The service will run the ONNX file, so check it agrees with PyTorch.
    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    pkg = next(i for i, n in model.names.items() if n == "package")
    worst = 0.0
    files = sorted((OUT / "val").glob("*/*.jpg"))
    for f in files:
        x = np.asarray(Image.open(f).convert("RGB").resize((IMGSZ, IMGSZ)), np.float32) / 255.0
        p_onnx = float(sess.run(None, {inp: x.transpose(2, 0, 1)[None]})[0][0][pkg])
        p_pt = float(model.predict(str(f), imgsz=IMGSZ, verbose=False)[0].probs.data[pkg])
        worst = max(worst, abs(p_onnx - p_pt))
    print(f"onnx -> {dest}  (max |onnx - torch| on {len(files)} val crops: {worst:.4f})")
    return 0


def evaluate(model, val_dir: Path, threshold: float = 0.5) -> dict:
    names = model.names  # {idx: class}
    pkg_idx = next(i for i, n in names.items() if n == "package")
    tp = fp = tn = fn = 0
    errors = []
    for cls in ("package", "empty"):
        files = sorted((val_dir / cls).glob("*.jpg"))
        for i in range(0, len(files), 64):
            batch = files[i:i + 64]
            for f, res in zip(batch, model.predict([str(p) for p in batch], imgsz=IMGSZ, verbose=False)):
                p = float(res.probs.data[pkg_idx])
                said = p >= threshold
                if cls == "package":
                    tp, fn = tp + said, fn + (not said)
                else:
                    fp, tn = fp + said, tn + (not said)
                if said != (cls == "package"):
                    errors.append({"file": f.name, "truth": cls, "p_package": round(p, 3)})
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    return {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall and round(recall, 3), "precision": precision and round(precision, 3),
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--camera", default="doorbell")
    f = sub.add_parser("fit")
    f.add_argument("--epochs", type=int, default=20)
    sub.add_parser("export")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return cmd_prepare(a.data, a.out, a.camera, a.seed)
    if a.cmd == "export":
        return cmd_export()
    return cmd_fit(a.out, a.epochs, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
