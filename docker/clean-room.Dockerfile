# Hermetic clean-room image for the U-01 gate.
#
# Builds a minimal environment that contains ONLY the published Arena wheel (plus
# its declared extras) — no source repo, no trainer, no checkpoints. The build
# context must contain exactly one `arena-*.whl`. The clean-room commands are then
# run against this image with `--network none`, so any hidden download fails.
#
# Build:  docker build -f docker/clean-room.Dockerfile -t arena-cleanroom <context-with-wheel>
# Run:    docker run --rm --network none -v <sandbox>:/work -w /work arena-cleanroom bash run_cleanroom.sh
FROM python:3.12-slim

# Install ONLY the wheel + runtime extras, the way a stranger would. Network is
# available at build time; the clean-room run itself uses `--network none`.
#
# Use the CPU-only PyTorch build: the default CUDA wheels pull multiple GB of
# nvidia-* packages that are useless for this CPU inference workload and can
# exhaust the builder's disk. The clean room only runs tiny MLP policies.
COPY arena-*.whl /tmp/
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && WHEEL="$(ls /tmp/arena-*.whl)" \
 && python -m pip install --no-cache-dir "${WHEEL}[torch,pettingzoo]" \
 && rm -f /tmp/arena-*.whl \
 && arena --help >/dev/null

# The recipient's working directory is mounted at run time and holds ONLY the
# received bundles, match.yaml, and the clean-room doc/script.
WORKDIR /work
CMD ["bash", "run_cleanroom.sh"]
