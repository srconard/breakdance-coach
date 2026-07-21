# IG Extract — nightly 3D extraction on the NAS

(Deployed copy lives at `/volume1/docker/AI-Agent-Home/ig-extract/README.md`; repo copy in `analyzer_3d/nas_deploy/`.)

Nightly job that converts tagged Instagram movement clips into 3D SMPL motion
data for the breakdance-coach reference library. Follow-through on the
2026-07-20/21 GVHMR calibration pilot (`output/ig_ingest/extract_pilot/PILOT-REPORT.md`).

## What runs

DSM Task Scheduler ID **20** ("IG Extract Nightly"), daily **04:10** →
`/volume1/docker/AI-Agent-Home/scripts/ig-extract-nas.sh` →
`docker run --rm ig-extract:latest` (python 3.11 + modal 1.3.4 + numpy 1.26.4;
host python 3.8 can't run the Modal client). Inside the container
`analyzer_3d.src.nas_extract`:

1. Reads `/data/ig-extract/extract_queue.json` (63 auto-extract clips; policy:
   `library_worthiness=yes` AND not partner/lift AND not montage-like; battle
   clips flagged `review`, parkour flagged `floor_plane_only`).
2. Skips clips already done on disk (results are the dedup state), takes the
   next N pending (default 20).
3. Per clip: GVHMR on Modal T4 (app `gvhmr-pose-estimation`, 15-min timeout,
   kill+log+continue) → motion cleanup → geometric contact signatures (CPU).
4. Writes `/data/ig-extract/results/<stem>/`:
   - `<stem>_smpl.pkl` — raw GVHMR result (kept so crash recovery / re-cleanup
     never re-pays the GPU)
   - `<stem>_smpl_clean.pkl` — cleaned, library schema
     (`smpl_params_global{body_pose,global_orient,transl,betas}` + `fps` +
     `mesh` + `cleanup` metrics)
   - `<stem>_contacts.json` — geometric_v1 contact signature document
   - `<stem>_metrics.json` — status, attempts, GPU wall seconds, cost
     (GPU-s × $0.000164), cleanup before/after, review/floor-plane flags
5. Appends per-clip lines + a dated `SUMMARY` line with cost telemetry to
   `/data/logs/ig-extract.log`. Wrapper/stdout log: `logs/ig-extract-run.log`.

No Blender, no viewer, no VLM judge on the NAS — those stay PC batch steps.

## Controls

- **Batch size**: `ig-extract/nightly-limit.conf` — single number, default 20
  when the file is absent. `echo 10 > .../nightly-limit.conf`.
- **Run now**: `sudo /usr/syno/bin/synoschedtask --run id=20 check_time=0`
- **Disable**: `sudo sed -i "s/^state=enabled/state=disabled/" /usr/syno/etc/synoschedule.d/root/20.task && sudo /usr/syno/bin/synoschedtask --sync`
  (re-enable: same with disabled→enabled). Task note in the vault:
  `AGENTS/system/scheduled-tasks/recurring/ig-extract-nas.md`.
- **Retry policy**: failed clips retry on later nights, give up after 3
  attempts (`attempts` in `<stem>_metrics.json`). Delete the clip's results
  dir to force a full redo; delete only `*_smpl_clean.pkl`/`*_contacts.json`
  to redo CPU stages against the saved raw pkl (no GPU spend).
- **Queue**: regenerate on the PC (`python analyzer_3d/src/ig_ingest.py
  extract-queue`), then
  `cat output/ig_ingest/extract_queue.json | ssh nas 'cat > /volume1/docker/AI-Agent-Home/ig-extract/extract_queue.json'`.

## Pulling results to the PC (no scp/SFTP on the NAS)

    ssh nas 'cd /volume1/docker/AI-Agent-Home/ig-extract && tar cf - results' | tar xf - -C output/ig_ingest/

## Rebuilding the image

Staged build dir: `ig-extract/build/` (Dockerfile + trimmed `analyzer_3d`
package: `nas_extract.py`, `motion_cleanup.py`, `contact_signatures.py`,
`contact_geometric.py`). After changing extractor code on the PC, re-stage the
changed files (tar-over-ssh) and:

    ssh nas 'cd /volume1/docker/AI-Agent-Home/ig-extract/build && sudo -n /usr/local/bin/docker build -t ig-extract:latest .'

## Secrets

`ig-extract/.secrets/modal.toml` (600, dir 700) — copy of the PC's
`~/.modal.toml`, mounted read-only at `/root/.modal.toml`. Outside git and the
vault. Refresh with `cat ~/.modal.toml | ssh nas 'cat > .../modal.toml'`.
