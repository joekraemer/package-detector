# package-detector image. Modelled on birdlisten's Dockerfile, minus the
# parts that belong to audio/BirdNET: no tflite, no libsndfile, Python 3.12
# (onnxruntime has 3.12 wheels; birdlisten is pinned to 3.11 for tflite).
# Runtime inference is onnxruntime only: no PyTorch, no Ultralytics in
# the deployed image. Training tooling lives elsewhere (docs/MODEL.md).
FROM python:3.12-slim

# ffmpeg pulls the one-frame RTSP snapshot; tzdata gives zoneinfo the
# IANA zones for TZ. Apt cache cleared in the same layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tzdata \
 && apt-get clean \
 && find /var/lib/apt/lists -mindepth 1 -delete

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Pinned requirements first so this layer caches across code-only changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN test -f packagedetector.py && test -f loop.py

# /data holds the SQLite db and evidence snapshots; compose mounts a volume.
# /models holds the ONNX weights; mount it, do not bake weights into the image.
RUN useradd --create-home --uid 10001 app && mkdir /data /models && chown -R app:app /app /data /models
USER app
VOLUME /data

# loop.py calls main() once per poll and sleeps LOOP_INTERVAL between polls.
CMD ["python", "loop.py"]
