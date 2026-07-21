#!/bin/sh
# Nightly IG-clip 3D extraction (GVHMR on Modal T4, cleanup + contacts on CPU).
# Deployed at /volume1/docker/AI-Agent-Home/scripts/ig-extract-nas.sh
# Runs from DSM Task Scheduler (root), 04:10 nightly.
# Docs: /volume1/docker/AI-Agent-Home/ig-extract/README.md
set -u
BASE=/volume1/docker/AI-Agent-Home
DOCKER=/usr/local/bin/docker
RUNLOG="$BASE/logs/ig-extract-run.log"

# Batch size: first line of nightly-limit.conf (digits only), default 20.
LIMIT=20
if [ -f "$BASE/ig-extract/nightly-limit.conf" ]; then
  CONF=$(head -1 "$BASE/ig-extract/nightly-limit.conf" | tr -cd '0-9')
  [ -n "$CONF" ] && LIMIT=$CONF
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ig-extract] wrapper start limit=$LIMIT" >> "$RUNLOG"
"$DOCKER" run --rm \
  -v "$BASE:/data" \
  -v "$BASE/ig-extract/.secrets/modal.toml:/root/.modal.toml:ro" \
  ig-extract:latest \
  --limit "$LIMIT" >> "$RUNLOG" 2>&1
rc=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ig-extract] wrapper exit rc=$rc" >> "$RUNLOG"
exit $rc
