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
| `modal_finetune.py` | Warm-start fine-tune on one clip with the Newton backend | Modal GPU — **blocked**: the workspace has no payment method, so Modal refuses every GPU (T4 included) |

Run from the repo root on the PC (Modal token there):

    modal run physics_dance/modal_soma_track.py --npz output/physics_dance/bake/mixamo_fw1.npz --name sixstep_fw1
    modal run physics_dance/modal_soma_track.py --name control_mini --control-steps 450

ProtoMotions pinned at `13de0cc`. The tracker is NVIDIA Open Model License (personal research use here).
The Mixamo six-step source stays local (`output/`, not committed): Mixamo files may not be redistributed.
