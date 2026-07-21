# ig-extract:latest — nightly IG-clip 3D extraction runner (Synology NAS)
# Build (on the NAS, from the staged build dir):
#   sudo -n /usr/local/bin/docker build -t ig-extract:latest .
# The host python is 3.8 (too old for the modal client) — hence this container.
# modal is pinned to the PC's client version; numpy <2 so pickles from the
# Modal server (numpy 1.23) and the PC pipeline stay cross-loadable.
FROM python:3.11-slim
RUN pip install --no-cache-dir modal==1.3.4 numpy==1.26.4
WORKDIR /app
COPY analyzer_3d /app/analyzer_3d
ENTRYPOINT ["python", "-u", "-m", "analyzer_3d.src.nas_extract"]
