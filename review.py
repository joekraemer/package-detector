#!/usr/bin/env python3
"""Rank captured frames by "something new in the scene" and label them.

Runs on the laptop against a local copy of the capture volume (see
docs/MODEL.md, "copy the frames out"). Two subcommands:

  python review.py score [--data package-data/frames]
      For each frame, build a reference image of the *empty* scene: the
      per-pixel median of baseline frames of the same camera taken at a
      similar time of day on OTHER days (packages are rare, so the median
      is almost always empty). Score = share of masked pixels that differ
      from that reference after matching overall brightness. The same is
      computed against the last baseline before the visit, and the
      smaller of the two is the score (dusk light and the DST shift only
      fool the first). Writes scores.jsonl next to the frames.

  python review.py serve [--port 8781]
      Loopback-only labelling page, highest score first. Keys:
      j/k or arrows move, p = package, e = empty, s = unsure, u = undo,
      r = flip the current card to its "before the visit" reference.
      /zones draws one rectangle per camera (saved to zones.json and
      re-scored); with a zone, only that rectangle counts.
      Labels append to labels.jsonl (last line per file wins).

The score is a ranking aid, not a label: a parked car, a person or a
shadow also scores high. A human (or later the VLM teacher) decides.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

DEFAULT_DATA = Path(__file__).resolve().parent / "package-data" / "frames"
SMALL = (160, 120)  # score at quarter resolution: fast, ignores JPEG noise
# Rows to ignore, as fractions of the image height: the timestamp overlay
# at the top changes every frame, and the garage's top strip is the street.
MASK_TOP = {"doorbell": 0.08, "garage": 0.30}
# Drawn on the /zones page: {camera: [x0, y0, x1, y1]} as fractions of the
# frame. When a camera has a zone, only that rectangle is scored. Committed
# to the repo because the classifier crops the same zone later.
ZONES_FILE = Path(__file__).resolve().parent / "zones.json"
DIFF_THRESHOLD = 30  # 0-255 grey levels, after brightness matching
REF_WINDOW_MIN = (60, 120, 240)  # widen until enough reference frames
REF_MIN_FRAMES = 3
LABELS = {"package", "empty", "unsure"}


# --------------------------------------------------------------- loading

def load_index(data: Path) -> list[dict]:
    """All index.jsonl rows whose image exists, oldest first."""
    rows = []
    for idx in sorted(data.glob("*/index.jsonl")):
        for line in idx.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (data / row["file"]).is_file():
                row["dt"] = datetime.fromisoformat(row["ts"])
                rows.append(row)
    rows.sort(key=lambda r: r["dt"])
    return rows


def grey(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(SMALL), dtype=np.float32)


def minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def circ_dist(a: int, b: int) -> int:
    d = abs(a - b) % 1440
    return min(d, 1440 - d)


# --------------------------------------------------------------- scoring

def reference_for(row: dict, baselines: list[dict], cache: dict) -> np.ndarray | None:
    """Median of same-camera baselines from other days near this time of day."""
    day = row["dt"].date()
    mod = minute_of_day(row["dt"])
    for window in REF_WINDOW_MIN:
        picks = [b for b in baselines
                 if b["dt"].date() != day and circ_dist(minute_of_day(b["dt"]), mod) <= window]
        if len(picks) >= REF_MIN_FRAMES:
            break
    else:
        return None
    key = tuple(b["file"] for b in picks)
    if key not in cache:
        cache[key] = np.median(np.stack([cache["img"](b["file"]) for b in picks]), axis=0)
    return cache[key]


PREV_MAX_GAP_MIN = 90


def event_start(row: dict) -> datetime:
    """When the visit began (event id is its start time), else the frame time."""
    ev = row.get("event")
    if ev:
        try:
            return datetime.strptime(ev, "%Y%m%dT%H%M%S").replace(tzinfo=row["dt"].tzinfo)
        except ValueError:
            pass
    return row["dt"]


def previous_baseline(row: dict, baselines: list[dict]) -> dict | None:
    """Last same-camera baseline strictly before this frame's visit began."""
    start = event_start(row)
    best = None
    for b in baselines:  # sorted oldest first
        if b["dt"] >= start or b["file"] == row["file"]:
            break
        best = b
    if best is None or (start - best["dt"]).total_seconds() > PREV_MAX_GAP_MIN * 60:
        return None
    return best


def load_zones(path: Path = ZONES_FILE) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {cam: z for cam, z in raw.items() if valid_zone(z)}


def valid_zone(z) -> bool:
    return (isinstance(z, list) and len(z) == 4
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in z)
            and z[2] - z[0] >= 0.02 and z[3] - z[1] >= 0.02)


def crop(img: np.ndarray, zone: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = zone
    return img[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]


def change_score(img: np.ndarray, ref: np.ndarray, camera: str,
                 zone: list[float] | None = None) -> float:
    if zone:
        a, b = crop(img, zone), crop(ref, zone)
    else:
        top = int(img.shape[0] * MASK_TOP.get(camera, 0.08))
        a, b = img[top:], ref[top:]
    # Match overall brightness so a cloud or dusk doesn't count as change.
    a = a * (b.mean() / max(a.mean(), 1.0))
    return float((np.abs(a - b) > DIFF_THRESHOLD).mean())


def cmd_score(data: Path, zones_path: Path = ZONES_FILE) -> int:
    zones = load_zones(zones_path)
    rows = load_index(data)
    if not rows:
        print(f"no frames under {data}", file=sys.stderr)
        return 1
    imgs: dict[str, np.ndarray] = {}

    def img(f: str) -> np.ndarray:
        if f not in imgs:
            imgs[f] = grey(data / f)
        return imgs[f]

    by_cam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["reason"] == "baseline":
            by_cam[r["camera"]].append(r)

    out = data / "scores.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")  # the page may read while we write
    n = 0
    with tmp.open("w") as fh:
        for cam_rows in _group(rows, "camera").values():
            cache: dict = {"img": img}
            bases = by_cam[cam_rows[0]["camera"]]
            for r in cam_rows:
                ref = reference_for(r, bases, cache)
                frame = img(r["file"])
                zone = zones.get(r["camera"])
                s_days = None if ref is None else change_score(frame, ref, r["camera"], zone)
                prev = previous_baseline(r, bases)
                s_prev = None if prev is None else change_score(frame, img(prev["file"]), r["camera"], zone)
                # A package left behind is new against both references; dusk
                # light or the clock change only against the other days.
                parts = [s for s in (s_days, s_prev) if s is not None]
                score = round(min(parts), 4) if parts else None
                fh.write(json.dumps({k: r.get(k) for k in ("file", "camera", "ts", "reason", "phase", "event")}
                                    | {"score": score,
                                       "vs_days": None if s_days is None else round(s_days, 4),
                                       "vs_prev": None if s_prev is None else round(s_prev, 4),
                                       "prev": prev and prev["file"]}) + "\n")
                n += 1
            # Free this camera's pixels before the next one.
            imgs.clear()
    tmp.replace(out)
    zoned = ", ".join(sorted(zones)) or "none"
    print(f"scored {n} frames -> {out} (zones: {zoned})")
    return 0


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


# --------------------------------------------------------------- labels

def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("label"):
                    labels[row["file"]] = row["label"]
                else:
                    labels.pop(row["file"], None)  # undo
    return labels


# --------------------------------------------------------------- server

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Package review</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#222;padding:8px 12px;display:flex;gap:12px;align-items:center;z-index:2}
header label{display:flex;gap:4px;align-items:center}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:8px;padding:8px}
.card{background:#1b1b1b;border:3px solid transparent;border-radius:6px;padding:4px;cursor:pointer}
.card.cur{border-color:#4ea1ff}
.card img{width:100%;display:block;border-radius:3px}
.meta{display:flex;justify-content:space-between;font-size:12px;padding:3px 2px;color:#bbb}
.lab-package{background:#173d1f}.lab-empty{background:#2a2a2a;opacity:.55}.lab-unsure{background:#3d3517}
.tag{font-weight:600;color:#fff}
kbd{background:#333;border-radius:3px;padding:0 4px}
</style></head><body>
<header>
 <strong>Package review</strong>
 <label>Camera <select id="cam"><option value="">all</option><option>doorbell</option><option>garage</option></select></label>
 <label>Reason <select id="reason"><option value="">all</option><option>baseline</option><option>arrive</option>
   <option>during_3s</option><option>during_7s</option><option>during_15s</option>
   <option>after_30s</option><option>after_120s</option><option>uncertain</option></select></label>
 <label><input type="checkbox" id="unlab" checked> unlabelled only</label>
 <span id="count"></span> <a href="zones" style="color:#4ea1ff">draw zones</a>
 <span style="margin-left:auto"><kbd>j</kbd>/<kbd>k</kbd> move · <kbd>p</kbd> package · <kbd>e</kbd> empty ·
 <kbd>s</kbd> unsure · <kbd>u</kbd> undo · <kbd>r</kbd> show/hide reference · click = full size</span>
</header>
<div id="grid" role="list"></div>
<script>
let all=[], view=[], cur=0, shown=60, showRef=false;
const grid=document.getElementById('grid');
async function load(){ all=await (await fetch('api/frames')).json(); render(); }
function filt(){
  const c=cam.value, r=reason.value, u=unlab.checked;
  view=all.filter(f=>(!c||f.camera===c)&&(!r||f.reason===r)&&(!u||!f.label));
}
function card(f,i){
  const d=document.createElement('div');
  d.className='card'+(i===cur?' cur':'')+(f.label?' lab-'+f.label:'');
  d.setAttribute('role','listitem');
  const src=(showRef&&i===cur&&f.prev)?f.prev:f.file;
  d.innerHTML=`<img loading="lazy" src="img/${encodeURI(src)}" alt="${f.camera} ${f.reason} ${f.ts}${src!==f.file?' (reference before visit)':''}">
   ${src!==f.file?'<div class="meta"><span class="tag">REFERENCE: baseline before this visit</span></div>':''}
   <div class="meta"><span>${f.ts.slice(5,16).replace('T',' ')} ${f.camera} ${f.reason}</span>
   <span>${f.label?'<span class="tag">'+f.label+'</span> ':''}score ${f.score==null?'–':f.score.toFixed(3)}</span></div>`;
  d.onclick=()=>{cur=i;window.open('img/'+encodeURI(f.file),'_blank');paint();};
  return d;
}
function render(){ filt(); cur=Math.min(cur,Math.max(view.length-1,0)); paint(); }
function paint(){
  grid.replaceChildren(...view.slice(0,Math.max(shown,cur+30)).map(card));
  document.getElementById('count').textContent=view.length+' frames · '+all.filter(f=>f.label==='package').length+' package';
  const el=grid.children[cur]; if(el) el.scrollIntoView({block:'nearest'});
}
async function setLabel(label){
  const f=view[cur]; if(!f) return;
  await fetch('api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file:f.file,label})});
  f.label=label||undefined;
  if(unlab.checked && label){ filt(); } else if(label){ cur=Math.min(cur+1,view.length-1); }
  paint();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT') return;
  const k=e.key;
  if(k==='j'||k==='ArrowRight'||k==='ArrowDown'){cur=Math.min(cur+1,view.length-1);paint();e.preventDefault();}
  else if(k==='k'||k==='ArrowLeft'||k==='ArrowUp'){cur=Math.max(cur-1,0);paint();e.preventDefault();}
  else if(k==='p') setLabel('package'); else if(k==='e') setLabel('empty');
  else if(k==='s') setLabel('unsure'); else if(k==='u') setLabel(null);
  else if(k==='r'){showRef=!showRef;paint();}
});
window.addEventListener('scroll',()=>{ if(innerHeight+scrollY>document.body.offsetHeight-800 && shown<view.length){shown+=60;paint();} });
[cam,reason,unlab].forEach(x=>x.onchange=()=>{cur=0;shown=60;render();});
load();
</script></body></html>"""


ZONES_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Draw zones</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#eee;padding:12px}
.cam{display:inline-block;vertical-align:top;margin:0 16px 16px 0}
.wrap{position:relative;width:640px;height:480px;cursor:crosshair;user-select:none;touch-action:none}
.wrap img{width:640px;height:480px;display:block}
.box{position:absolute;border:3px solid #ffd400;background:rgba(255,212,0,.15);pointer-events:none}
button{font:inherit;padding:6px 14px;margin-right:8px}
#status{margin-left:8px}
code{color:#ffd400}
</style></head><body>
<h1 style="font-size:18px;margin:0 0 8px">Draw one rectangle per camera where packages land</h1>
<p style="margin:0 0 12px;color:#bbb">Drag on a picture to draw. Keep it tight, but include the
spots a courier might use (edge of the mat, beside the table). Then <b>Save &amp; re-score</b>.
<a href="./" style="color:#4ea1ff">Back to review</a></p>
<div id="cams"></div>
<button id="save">Save &amp; re-score</button><span id="status" role="status"></span>
<script>
let zones={};
async function init(){
  const r=await (await fetch('api/zones')).json(); zones=r.zones;
  const host=document.getElementById('cams');
  for(const [cam,file] of Object.entries(r.samples)){
    const div=document.createElement('div'); div.className='cam';
    div.innerHTML=`<h2 style="font-size:15px;margin:0 0 4px">${cam} <code id="z-${cam}"></code>
      <button data-clear="${cam}" style="font-size:12px;padding:2px 8px">clear</button></h2>
      <div class="wrap" id="w-${cam}"><img src="img/${encodeURI(file)}" alt="${cam} sample frame" draggable="false"><div class="box" id="b-${cam}"></div></div>`;
    host.appendChild(div); wire(cam); show(cam);
  }
  document.querySelectorAll('[data-clear]').forEach(b=>b.onclick=()=>{delete zones[b.dataset.clear];show(b.dataset.clear);});
}
function show(cam){
  const b=document.getElementById('b-'+cam), z=zones[cam], t=document.getElementById('z-'+cam);
  if(!z){b.style.display='none'; t.textContent='(no zone)'; return;}
  Object.assign(b.style,{display:'block',left:z[0]*100+'%',top:z[1]*100+'%',width:(z[2]-z[0])*100+'%',height:(z[3]-z[1])*100+'%'});
  t.textContent=z.map(v=>v.toFixed(2)).join(', ');
}
function wire(cam){
  const w=document.getElementById('w-'+cam); let start=null;
  const pos=e=>{const r=w.getBoundingClientRect();return [Math.min(Math.max((e.clientX-r.left)/r.width,0),1),Math.min(Math.max((e.clientY-r.top)/r.height,0),1)];};
  w.onpointerdown=e=>{start=pos(e); w.setPointerCapture(e.pointerId);};
  w.onpointermove=e=>{ if(!start) return; const p=pos(e);
    zones[cam]=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])]; show(cam);};
  w.onpointerup=()=>{start=null;};
}
document.getElementById('save').onclick=async()=>{
  const s=document.getElementById('status'); s.textContent='saving and re-scoring (about 20 s)…';
  const r=await fetch('api/zones',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(zones)});
  s.textContent=r.ok?'saved. Re-scored with the new zones.':'save failed ('+r.status+')';
};
init();
</script></body></html>"""


def sample_frames(data: Path) -> dict[str, str]:
    """Latest daytime baseline per camera, for drawing zones on."""
    out: dict[str, str] = {}
    scores = data / "scores.jsonl"
    if scores.is_file():
        for line in scores.read_text().splitlines():
            r = json.loads(line)
            if r["reason"] == "baseline" and r["phase"] == "day":
                out[r["camera"]] = r["file"]  # file is time-ordered per camera
    return out


def make_handler(data: Path, port: int, zones_path: Path = ZONES_FILE, on_label=None):
    labels_path = data / "labels.jsonl"
    score_lock = threading.Lock()
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    root = data.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _host_ok(self) -> bool:
            # Loopback bind + Host check stops DNS-rebinding pages from
            # reading frames or writing labels.
            if self.headers.get("Host") not in allowed_hosts:
                self.send_error(403)
                return False
            return True

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/zones":
                return self._send(200, ZONES_PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/zones":
                body = {"zones": load_zones(zones_path), "samples": sample_frames(data)}
                return self._send(200, json.dumps(body).encode(), "application/json")
            if path == "/api/frames":
                labels = read_labels(labels_path)
                rows = []
                scores = data / "scores.jsonl"
                if scores.is_file():
                    for line in scores.read_text().splitlines():
                        r = json.loads(line)
                        if r["file"] in labels:
                            r["label"] = labels[r["file"]]
                        rows.append(r)
                rows.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1))
                return self._send(200, json.dumps(rows).encode(), "application/json")
            if path.startswith("/img/"):
                target = (root / unquote(path[5:])).resolve()
                if root not in target.parents or target.suffix != ".jpg" or not target.is_file():
                    return self.send_error(404)
                return self._send(200, target.read_bytes(), "image/jpeg")
            self.send_error(404)

        def do_POST(self):
            if not self._host_ok():
                return
            path = urlparse(self.path).path
            if self.headers.get("Content-Type") != "application/json":
                return self.send_error(400)
            if path == "/api/zones":
                return self._save_zones()
            if path != "/api/label":
                return self.send_error(400)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))[:10_000])
                f, label = str(body["file"]), body.get("label")
            except (ValueError, KeyError, TypeError):
                return self.send_error(400)
            if label is not None and label not in LABELS:
                return self.send_error(400)
            if not (root / f).resolve().is_file():
                return self.send_error(404)
            with labels_path.open("a") as fh:
                fh.write(json.dumps({"file": f, "label": label,
                                     "at": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
            if on_label is not None:
                on_label()  # schedules the debounced data backup; returns at once
            self._send(200, b"{}", "application/json")

        def _save_zones(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))[:10_000])
            except ValueError:
                return self.send_error(400)
            if not isinstance(body, dict) or not all(
                    isinstance(c, str) and c.isidentifier() and valid_zone(z) for c, z in body.items()):
                return self.send_error(400)
            zones = {c: [round(float(v), 3) for v in z] for c, z in sorted(body.items())}
            zones_path.write_text(json.dumps(zones, indent=2) + "\n")
            with score_lock:  # one re-score at a time
                cmd_score(data, zones_path)
            self._send(200, json.dumps(zones).encode(), "application/json")

    return Handler


def cmd_serve(data: Path, port: int) -> int:
    if not (data / "scores.jsonl").is_file():
        print("run `python review.py score` first", file=sys.stderr)
        return 1
    import datasync
    # Labels are backed up to the private data repo 2 min after the last
    # one, and on exit. Only for the checkout's own package-data: that is
    # what the data repo's sync.py reads.
    saver = None
    if data.resolve() == DEFAULT_DATA.resolve():
        saver = datasync.Debouncer(datasync.backup, delay_s=120, log=lambda m: print(m, flush=True))
    srv = ThreadingHTTPServer(("127.0.0.1", port),
                              make_handler(data, port, on_label=saver.schedule if saver else None))
    print(f"review page: http://127.0.0.1:{port}/  (labels -> {data / 'labels.jsonl'})")
    print(datasync.backup() if saver else "data backup off: --data is not this checkout's package-data")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if saver:
            saver.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="local copy of the frames dir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score")
    sp = sub.add_parser("serve")
    sp.add_argument("--port", type=int, default=8781)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        return cmd_score(a.data)
    return cmd_serve(a.data, a.port)


if __name__ == "__main__":
    raise SystemExit(main())
