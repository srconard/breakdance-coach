# physics_dance — moves into a pretrained physics body (2026-11-17)

First physical-dance tests (vault: `AGENTS/desks/movement-simulation/docs/2026-11-17 six-step-physics-test.md`).

| File | What it does | Where it runs |
|---|---|---|
| `inspect_sixstep.py` | Scores skeleton bakes for the six-step signature (feet circling the support hand) + contact sheets | PC CPU |
| `feasibility.py` | Experiment 1: kinematic physics screen (floating, sliding, jolts, balance) -> trackable-now / needs-cleaning / broken | PC CPU |
| `bake_from_pkl.py` | GVHMR clean pkl -> bake npz at 30 fps | PC CPU |
| `soma_from_bake.py` | Bake npz -> ProtoMotions SOMA23 `.motion` (Y-up/+Z -> Z-up/-Y basis change, floor fit, `--speed`, `--lift`) | Modal container |
| `track_record.py` | Runs a ProtoMotions tracker headless in MuJoCo and records sim + reference states | Modal container |
| `render_compare.py` | Side-by-side video (blue reference, orange physics) + metrics | Modal container |
| `modal_soma_track.py` | The whole chain on one Modal **CPU** container: convert -> anatomical-SOMA retarget (PyRoki/JAX) -> NVIDIA anatomical SOMA FSQ tracker (MuJoCo) -> video | Modal CPU |
| `prepare_batch.py` | *(2026-11-24)* Bakes every Move Library motion that can reach the body (Blender for Mixamo FBX / CMU BVH, `bake_from_pkl` for SMPL pkls) into `output/physics_dance/bake_lib/` and writes the Modal manifests `output/physics_dance/manifest_<group>.json` (`ig_trackable`, `mocap`, `ig_usable`). Skips bakes that already exist: delete a bake to redo it | PC CPU (Blender 4.4) |
| `check_bake_rest.py` | *(2026-11-24)* Sanity check of a bake before any Modal spend: rest facing, rest limb directions, frame 0 vs rest, per-frame jumps of the first frames | PC CPU |
| `verdicts.py` | *(2026-11-24)* Reads every `output/physics_dance/runs/<name>/metrics.json` + the manifests and writes `output/physics_dance/library/runs.json` + one side-by-side mp4 per run, with the three labels (holds up / partly / falls apart; thresholds in its docstring). `NOT_A_VERDICT` / `NOT_RUN_REASON` list runs that must not be shown as a verdict | PC CPU (stdlib) |
| `modal_finetune.py` | Warm-start fine-tune on one clip with the Newton backend | Modal GPU — **blocked**: the workspace has no payment method, so Modal refuses every GPU (T4 included) |

Run from the repo root on the PC (Modal token there):

    modal run physics_dance/modal_soma_track.py --npz output/physics_dance/bake/mixamo_fw1.npz --name sixstep_fw1
    modal run physics_dance/modal_soma_track.py --name control_mini --control-steps 450

ProtoMotions pinned at `13de0cc`. The tracker is NVIDIA Open Model License (personal research use here).
The Mixamo six-step source stays local (`output/`, not committed): Mixamo files may not be redistributed.

## Physics check on the Move Library: rebuild order (2026-11-24)

Each step on the PC runs from the repo root (`pc-run.sh breakdance-coach '<cmd>' --if-hot 0` from the NAS). Set `PYTHONUTF8=1` before `modal run` on Windows, or its progress output crashes on a cp1252 console.

1. `python -m physics_dance.prepare_batch`: bakes + manifests. Then `python -m physics_dance.check_bake_rest <bake.npz>` on any new source kind.
2. `modal run physics_dance/modal_soma_track.py::batch --manifest output/physics_dance/manifest_<group>.json [--only a,b]`: Modal CPU, about $0.02-0.03 per clip. A run whose `runs/<name>/metrics.json` exists is skipped (never paid twice), so to redo a run move its folder out of `runs/` first (we keep them in `output/physics_dance/runs_superseded/`).
3. `modal run physics_dance/modal_soma_track.py::rescore --names a,b`: metrics only, from saved recordings (no re-simulation).
4. `python -m physics_dance.verdicts`: writes `output/physics_dance/library/`.
5. Copy `output/physics_dance/library/*` to the NAS `ig-extract/library-assets/physics/`, then rebuild the site on the NAS (`Docs/move-library.md`).

### Source skeletons: what the body assumes (2026-11-24)

`soma_from_bake.py` copies each bone's rotation relative to the source REST onto SOMA23, whose rest is a straight-limbed T-pose. Two CMU faults broke this, both fixed on 2026-11-24:

- the cgspeed CMU BVHs open on a T-pose frame facing another way (a ~175 deg whole-body snap at frame 0 -> 1). `skeleton_bake_npz.py` now drops leading frames where any bone turns more than 90 deg in one sample;
- CMU femurs and shins rest ~20 deg out (the ASF femur direction), so every pose came out with the legs crossed. `soma_from_bake.py` (`rest_fix`) now post-multiplies a limb's rotation by the canonical -> source rest rotation when its rest is more than 10 deg off the T-pose (it prints `rest_fix_deg`; Mixamo and SMPL bakes stay under 10 deg and are unchanged).
