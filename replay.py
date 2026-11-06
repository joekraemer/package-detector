#!/usr/bin/env python3
"""replay.py: run the alert rules over the detector's poll log.

The detector logs p(package) for every poll (~4,000 a day per camera) in
its SQLite db. Frame-level validation cannot say how often the phone
would buzz; this can. For each cutoff it replays the service's own state
machine (packagedetector.update_state: CONFIRM_HITS polls to switch on,
CLEAR_MISSES to switch off, a notification per arrival outside the
cooldown) and checks every alert against the labelled frames:

- an alert is TRUE if a frame labelled "package" was taken within
  -10/+30 min of it, FALSE if only "empty" frames were, and UNVERIFIED if
  no labelled frame is near (label one on the review page);
- a delivery is a run of "package" frames; it counts as CAUGHT if the
  replayed sensor was on at some point during it.

It also lists the UNCERTAIN stretches, runs of polls whose score sat in
the uncertain band, grouped by light (day/twilight/night from the nearest
capture frame) and hour: the situations that need more training data.

The poll log only holds the deployed model's scores, so this measures the
model that was running at the time. To compare a new model, use
`train.py compare` on the fixed test days.

Usage (laptop, after copying the detector's data, see docs/MODEL.md):
    python replay.py                                   # doorbell, cutoffs 0.5-0.9
    python replay.py --cutoff 0.7 --cutoff 0.8 --since 2026-11-05
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import packagedetector
import review
import train

REPO = Path(__file__).resolve().parent
DB = REPO / "package-data" / "detector" / "package-detector.sqlite"
TRUE_BEFORE, TRUE_AFTER = timedelta(minutes=10), timedelta(minutes=30)
EMPTY_NEAR = timedelta(minutes=30)
DELIVERY_GAP = timedelta(minutes=45)  # package frames further apart = two deliveries
STRETCH_GAP = timedelta(seconds=90)   # polls further apart break an uncertain stretch


@dataclass
class Alert:
    at: datetime
    p: float
    verdict: str = ""        # true | false | unverified
    evidence: str = ""       # the nearest labelled frame


@dataclass
class Replay:
    cutoff: float
    alerts: list[Alert] = field(default_factory=list)
    switches_on: list[datetime] = field(default_factory=list)
    on_polls: list[datetime] = field(default_factory=list)


def load_polls(db: Path, camera: str, since: datetime | None = None) -> list[tuple[datetime, float]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT ts, confidence FROM polls WHERE camera=? AND confidence IS NOT NULL "
                            "ORDER BY ts", (camera,)).fetchall()
    finally:
        conn.close()
    polls = [(datetime.fromisoformat(ts), float(p)) for ts, p in rows]
    return [x for x in polls if since is None or x[0] >= since]


def replay(polls: list[tuple[datetime, float]], cutoff: float, hits: int, misses: int,
           cooldown_s: int) -> Replay:
    """The service's rules, poll by poll. A notification goes out on each
    off->on switch unless the last one was under cooldown_s ago."""
    out = Replay(cutoff)
    present, h, m = False, 0, 0
    last_notified: datetime | None = None
    for at, p in polls:
        present, h, m, changed = packagedetector.update_state(present, h, m, p >= cutoff, hits, misses)
        if present:
            out.on_polls.append(at)
        if changed and present:
            out.switches_on.append(at)
            if last_notified is None or (at - last_notified).total_seconds() >= cooldown_s:
                out.alerts.append(Alert(at, p))
                last_notified = at
    return out


def frame_truth(data: Path, camera: str) -> tuple[list[tuple[datetime, str, str]], list[dict]]:
    """[(time, label, file)] for every labelled frame of the camera, sorted,
    plus the camera's index rows (for light/phase lookups)."""
    rows = [r for r in review.load_index(data) if r["camera"] == camera]
    staged = {s for s in train.read_staged_days(data / "labels.jsonl") if s[0] == camera}
    labels = train.resolve_labels(rows, review.read_labels(data / "labels.jsonl"), staged)
    by_file = {r["file"]: r for r in rows}
    truth = sorted((by_file[f]["dt"], l, f) for f, l in labels.items() if f in by_file)
    return truth, rows


def judge(alerts: list[Alert], truth: list[tuple[datetime, str, str]]) -> None:
    times = [t for t, _, _ in truth]
    for a in alerts:
        lo, hi = bisect_left(times, a.at - TRUE_BEFORE), bisect_right(times, a.at + TRUE_AFTER)
        pkg = [x for x in truth[lo:hi] if x[1] == "package"]
        if pkg:
            a.verdict, a.evidence = "true", pkg[0][2]
            continue
        lo, hi = bisect_left(times, a.at - EMPTY_NEAR), bisect_right(times, a.at + EMPTY_NEAR)
        near = truth[lo:hi]
        if near:
            a.verdict = "false"
            a.evidence = min(near, key=lambda x: abs(x[0] - a.at))[2]
        else:
            a.verdict = "unverified"


def deliveries(truth: list[tuple[datetime, str, str]], start: datetime, end: datetime
               ) -> list[tuple[datetime, datetime, int]]:
    """Runs of 'package' frames inside the polled period: (first, last, n)."""
    out: list[list] = []
    for t, label, _ in truth:
        if label != "package" or not start <= t <= end:
            continue
        if out and t - out[-1][1] <= DELIVERY_GAP:
            out[-1][1], out[-1][2] = t, out[-1][2] + 1
        else:
            out.append([t, t, 1])
    return [tuple(x) for x in out]


def caught(r: Replay, first: datetime, last: datetime) -> bool:
    i = bisect_left(r.on_polls, first - timedelta(minutes=2))
    return i < len(r.on_polls) and r.on_polls[i] <= last + timedelta(minutes=5)


def light_at(rows: list[dict], at: datetime) -> str:
    """day / twilight / night from the nearest capture frame that has a phase."""
    phased = [r for r in rows if r.get("phase") in ("day", "night", "twilight")]
    if not phased:
        return "unknown"
    times = [r["dt"] for r in phased]
    i = bisect_left(times, at)
    cands = [phased[j] for j in (i - 1, i) if 0 <= j < len(phased)]
    return min(cands, key=lambda r: abs(r["dt"] - at))["phase"]


def uncertain_stretches(polls: list[tuple[datetime, float]], low: float, high: float,
                        min_polls: int) -> list[tuple[datetime, datetime, int, float]]:
    """(start, end, n_polls, max_p) for runs of >= min_polls polls in [low, high)."""
    out, run = [], []
    for at, p in polls:
        if low <= p < high and (not run or at - run[-1][0] <= STRETCH_GAP):
            run.append((at, p))
            continue
        if len(run) >= min_polls:
            out.append((run[0][0], run[-1][0], len(run), max(x[1] for x in run)))
        run = [(at, p)] if low <= p < high else []
    if len(run) >= min_polls:
        out.append((run[0][0], run[-1][0], len(run), max(x[1] for x in run)))
    return out


def local(t: datetime) -> str:
    return t.astimezone().strftime("%m-%d %H:%M")


def run(args) -> dict:
    since = datetime.fromisoformat(args.since).astimezone() if args.since else None
    polls = load_polls(args.db, args.camera, since)
    if not polls:
        raise SystemExit(f"no polls for {args.camera} in {args.db}")
    truth, rows = frame_truth(args.data, args.camera)
    start, end = polls[0][0], polls[-1][0]
    days = max((end - start).total_seconds() / 86400, 1e-9)
    dels = deliveries(truth, start, end)
    report = {"camera": args.camera, "polls": len(polls), "from": start.isoformat(), "to": end.isoformat(),
              "days": round(days, 2), "deliveries": [(a.isoformat(), b.isoformat(), n) for a, b, n in dels],
              "cutoffs": {}}
    for cutoff in args.cutoff:
        r = replay(polls, cutoff, args.hits, args.misses, args.cooldown)
        judge(r.alerts, truth)
        v = Counter(a.verdict for a in r.alerts)
        report["cutoffs"][cutoff] = {
            "alerts": len(r.alerts), "true": v["true"], "false": v["false"], "unverified": v["unverified"],
            "false_per_day": round(v["false"] / days, 2),
            "switches_on": len(r.switches_on),
            "deliveries_caught": sum(caught(r, a, b) for a, b, _ in dels),
            "alert_list": [(a.at.isoformat(), round(a.p, 3), a.verdict, a.evidence) for a in r.alerts]}
    stretches = uncertain_stretches(polls, args.band[0], args.band[1], args.min_polls)
    report["uncertain"] = [(a.isoformat(), b.isoformat(), n, round(p, 3), light_at(rows, a))
                           for a, b, n, p in stretches]
    return report


def print_report(rep: dict, show: int) -> None:
    start, end = datetime.fromisoformat(rep["from"]), datetime.fromisoformat(rep["to"])
    print(f"{rep['camera']}: {rep['polls']} polls, {local(start)} -> {local(end)} ({rep['days']} days)")
    dels = rep["deliveries"]
    print(f"deliveries in labelled frames: {len(dels)}"
          + "".join(f"\n  {local(datetime.fromisoformat(a))}-{local(datetime.fromisoformat(b))[6:]} "
                    f"({n} frames)" for a, b, n in dels))
    print("\ncutoff  alerts  true  false  unverified  false/day  deliveries caught")
    for c, r in rep["cutoffs"].items():
        print(f"{float(c):>6}  {r['alerts']:>6}  {r['true']:>4}  {r['false']:>5}  {r['unverified']:>10}"
              f"  {r['false_per_day']:>9}  {r['deliveries_caught']}/{len(dels)}")
    for c, r in rep["cutoffs"].items():
        bad = [a for a in r["alert_list"] if a[2] != "true"]
        if bad:
            print(f"\nnot-true alerts at {c}:")
            for at, p, verdict, ev in bad[:show]:
                print(f"  {local(datetime.fromisoformat(at))}  p={p}  {verdict:<10}  {ev}")
    un = rep["uncertain"]
    print(f"\nuncertain stretches: {len(un)}")
    by_light = Counter(x[4] for x in un)
    by_hour = Counter(datetime.fromisoformat(x[0]).astimezone().hour for x in un)
    print("  by light: " + ", ".join(f"{k} {v}" for k, v in by_light.most_common()))
    print("  by hour:  " + ", ".join(f"{h:02d}h {n}" for h, n in sorted(by_hour.items())))
    for a, b, n, p, light in sorted(un, key=lambda x: -x[2])[:show]:
        print(f"  {local(datetime.fromisoformat(a))}-{local(datetime.fromisoformat(b))[6:]}  "
              f"{n:>3} polls  max p {p}  {light}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--data", type=Path, default=train.DATA)
    ap.add_argument("--camera", default="doorbell")
    ap.add_argument("--cutoff", type=float, action="append")
    ap.add_argument("--hits", type=int, default=2, help="CONFIRM_HITS")
    ap.add_argument("--misses", type=int, default=5, help="CLEAR_MISSES")
    ap.add_argument("--cooldown", type=int, default=1800, help="HA_NOTIFY_COOLDOWN_SECONDS")
    ap.add_argument("--band", type=float, nargs=2, default=(0.4, 0.8), metavar=("LOW", "HIGH"))
    ap.add_argument("--min-polls", type=int, default=3, help="shortest uncertain stretch reported")
    ap.add_argument("--since", help="ISO date/time; only polls from then on")
    ap.add_argument("--show", type=int, default=15, help="list at most this many items per section")
    ap.add_argument("--json", type=Path, help="also write the full report here")
    args = ap.parse_args(argv)
    args.cutoff = sorted(args.cutoff or (0.5, 0.6, 0.7, 0.8, 0.9))
    rep = run(args)
    print_report(rep, args.show)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
