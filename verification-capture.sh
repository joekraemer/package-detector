#!/usr/bin/env bash
# Issue #2 verification capture: one ffmpeg snapshot per camera, saved
# locally, with a log that never contains credentials.
#
# Usage (from the package-detector repo root, after `cp .env.example .env`
# and filling in CAMERAS in .env -- never paste .env contents in chat):
#   ./verification-capture.sh            # one round, now
#   ./verification-capture.sh 40 15      # 40 rounds, 15 s apart (~10 min:
#                                        # the NVR session-churn test)
#
# Output: verification/<timestamp>_<camera>.jpg
#         verification/capture.log  (camera, time, ok/fail, bytes, seconds;
#                                    host only, no user/password in the log)
set -u
cd "$(dirname "$0")"

if [ ! -f .env ]; then echo "no .env found; copy .env.example to .env first" >&2; exit 2; fi
# shellcheck disable=SC1091
set -a; . ./.env; set +a
if [ -z "${CAMERAS:-}" ]; then echo "CAMERAS is empty in .env" >&2; exit 2; fi
if ! command -v ffmpeg >/dev/null 2>&1; then echo "ffmpeg not found" >&2; exit 2; fi

ROUNDS="${1:-1}"
SLEEP_S="${2:-0}"
mkdir -p verification
LOG=verification/capture.log

# Strip credentials for logging: rtsp://user:pass@host:port/... -> host
host_of() { printf '%s' "$1" | sed -E 's#^rtsp://[^@]*@##; s#/.*$##'; }

round=1
while [ "$round" -le "$ROUNDS" ]; do
  stamp=$(date +%Y-%m-%d_%H%M%S)
  IFS=',' read -r -a pairs <<< "$CAMERAS"
  for pair in "${pairs[@]}"; do
    name="${pair%%=*}"
    url="${pair#*=}"
    out="verification/${stamp}_${name}.jpg"
    start=$(date +%s)
    if ffmpeg -hide_banner -loglevel error -rtsp_transport tcp -i "$url" \
         -frames:v 1 -q:v 2 -y "$out" 2>/tmp/pd-cap.err; then
      bytes=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo "?")
      echo "$stamp round=$round camera=$name host=$(host_of "$url") ok bytes=$bytes secs=$(( $(date +%s) - start ))" >> "$LOG"
    else
      rm -f "$out"
      echo "$stamp round=$round camera=$name host=$(host_of "$url") FAIL $(tail -1 /tmp/pd-cap.err 2>/dev/null)" >> "$LOG"
    fi
  done
  round=$((round + 1))
  if [ "$round" -le "$ROUNDS" ] && [ "$SLEEP_S" != "0" ]; then sleep "$SLEEP_S"; fi
done
echo "done. Snapshots in verification/, log in $LOG"
echo "Failure summary:"
grep -c " ok " "$LOG" 2>/dev/null | xargs -I{} echo "  ok lines: {}"
grep " FAIL" "$LOG" 2>/dev/null | sed -E 's/.*camera=([^ ]+).*/  fail: \1/' | sort | uniq -c || true
