"""DART (DartControl, ICLR 2025) motion generation on Modal cloud GPU.

Rehearsal of the movement-generation flagship: reproduce DART's waypoint
(goal-reach) generation, then bend it around scanned scene geometry with a
mesh-penetration (SDF) loss at sampling time -- no retraining.

Axis convention: DART / AMASS are Z-UP with floor at z=0. Our test_scene.glb
is glTF Y-UP (1 unit = 1 m, ground top at y=0). prep_scene converts Y-up ->
Z-up explicitly via (x, y, z) -> (x, -z, y). All waypoints/goals in this file
are stated in the Z-UP DART world.

Pipeline (run in order):
    1. modal run analyzer_3d/src/dart_modal.py::download_assets
       (fetches DART's released checkpoints + data bundle from Google Drive)
    2. modal volume put dart-assets inbox/smplx_lockedhead_20230207.zip \
           /smplx_lockedhead_20230207.zip
       modal run analyzer_3d/src/dart_modal.py::setup_body_models
       (SMPL-X models: Shawn's registered download, uploaded -- never scraped)
    3. modal run analyzer_3d/src/dart_modal.py::prep_scene
       (test_scene.glb -> Z-up scene_with_floor.obj + mesh2sdf 256^3 grid)
    4. modal run analyzer_3d/src/dart_modal.py::stock_rollout
       (milestone: stock text-to-motion demo, proof the model runs)
    5. modal run analyzer_3d/src/dart_modal.py::experiment
       (run (a) stock waypoint vs (b) waypoint + SDF collision loss;
        straight path start->goal passes through the big crate at
        Z-up (3.5, 2.5); success = (b) visibly routes around/over it)

Outputs land in C:/dev/breakdance-coach/output/dart_rehearsal/.
"""

import io
import json
from pathlib import Path

import modal

app = modal.App("dart-motion-gen")

# Persistent volume: gdrive bundle (checkpoints + data), body models, scenes
assets_volume = modal.Volume.from_name("dart-assets", create_if_missing=True)

DART_SHA = "bb67ae1ed6ce051080468bf15bc6e54a6c3f8417"  # verified 2026-07-09
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps"

# Local paths (client side). Inside the Modal container this file lives at
# /root/dart_modal.py (no parents[2]) -- these paths are only used client-side.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    REPO_ROOT = Path("/root")
SCENE_GLB = REPO_ROOT / "output" / "reference_library" / "qa" / "test_scene.glb"
OUT_DIR = REPO_ROOT / "output" / "dart_rehearsal"

# ---------------------------------------------------------------------------
# Experiment geometry (Z-UP DART world, converted from the Y-up GLB):
#   big crate: 1m cube centered (3.5, 2.5, 0.5) -> occupies [3,4]x[2,3]x[0,1]
#   start pelvis xy (1.0, 0.5); goal pelvis xy (6.0, 4.5)
#   straight line start->goal passes exactly through crate center at t=0.5
# ---------------------------------------------------------------------------
START_XY = (1.0, 0.5)
GOAL_XY = (6.0, 4.5)
CRATE_BOX = ((3.0, 4.0), (2.0, 3.0), (0.0, 1.0))  # x, y, z ranges (Z-up)
TEXT_PROMPT = "walk*24"  # 24 primitives x 8 frames @30fps = 6.4s for ~6.4m

CKPT = "./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt"

dart_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl",
    )
    .pip_install(
        "numpy==1.23.5",
        "smplx==0.1.28",
        "trimesh",
        "tyro",
        "omegaconf",
        "tensorboard",
        "tqdm",
        "spacy",
        "click",
        "torch-dct",
        "loralib",
        "ftfy",
        "regex",
        "pyyaml",
        "matplotlib",
        "mesh2sdf",
        "scikit-image",
        "scipy",
        "gdown",
        "git+https://github.com/openai/CLIP.git",
    )
    .run_commands(
        f"git clone https://github.com/zkf1997/DART /DART && cd /DART && git checkout {DART_SHA}",
    )
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "PYTHONPATH": "/DART"})
)


# ---------------------------------------------------------------------------
# Shared helpers (run inside containers)
# ---------------------------------------------------------------------------

def _tree(root, max_depth=4, max_entries=400):
    """Compact directory tree with file sizes, for layout discovery."""
    import os

    lines = []
    root = str(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{os.path.basename(dirpath) or dirpath}/")
        for fn in sorted(filenames):
            size = os.path.getsize(os.path.join(dirpath, fn))
            lines.append(f"{indent}  {fn} ({size / 1e6:.1f} MB)")
            if len(lines) > max_entries:
                lines.append("... (truncated)")
                return "\n".join(lines)
    return "\n".join(lines)


def _merge_tree(src, dst):
    """Symlink-merge src dir into dst dir (recursing where both are dirs)."""
    import os

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if not target.exists():
            os.symlink(child, target)
        elif child.is_dir() and target.is_dir() and not target.is_symlink():
            _merge_tree(child, target)


def _assemble_workspace():
    """Merge the volume assets into the /DART checkout (repo files win)."""
    bundle = Path("/vol/bundle")
    if not bundle.exists():
        raise RuntimeError("Volume missing /bundle -- run download_assets first")
    # The gdrive bundle mirrors the repo layout (data/, mld_denoiser/, ...)
    _merge_tree(bundle, Path("/DART"))
    # Body models (uploaded separately, license-gated)
    bm = Path("/vol/body_models/smplx_lockedhead_20230207")
    if bm.exists():
        _merge_tree(bm.parent, Path("/DART/data"))


def _calc_sdf_cpu(sdf_grid, sdf_config, points):
    """Trilinear SDF lookup, mirroring DART's calc_point_sdf (torch CPU)."""
    import torch
    import torch.nn.functional as F

    grid = torch.as_tensor(sdf_grid, dtype=torch.float32)[None]  # [1,S,S,S]
    pts = torch.as_tensor(points, dtype=torch.float32)[None]  # [1,P,3]
    scale = float(sdf_config["scale"])
    center = torch.tensor(sdf_config["center"], dtype=torch.float32).view(1, 1, 3)
    p = (pts - center) * scale
    vals = F.grid_sample(
        grid.unsqueeze(0),
        p[:, :, [2, 1, 0]].view(1, -1, 1, 1, 3),
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)
    return (vals / scale).numpy()


# ---------------------------------------------------------------------------
# Step 1: download DART's released checkpoints + data bundle
# ---------------------------------------------------------------------------

@app.function(image=dart_image, volumes={"/vol": assets_volume}, timeout=3600)
def download_assets_remote(force: bool = False) -> str:
    import shutil
    import subprocess

    bundle = Path("/vol/bundle")
    if bundle.exists() and any(bundle.iterdir()) and not force:
        return "ALREADY PRESENT:\n" + _tree(bundle)

    tmp = Path("/tmp/gdrive")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    proc = subprocess.run(
        ["gdown", "--folder", GDRIVE_FOLDER, "-O", str(tmp)],
        capture_output=True, text=True,
    )
    log = proc.stdout[-3000:] + "\n" + proc.stderr[-3000:]
    if proc.returncode != 0 and not any(tmp.rglob("*")):
        return f"GDOWN FAILED (rc={proc.returncode}):\n{log}"

    # Extract any archives in place
    for arch in list(tmp.rglob("*.zip")) + list(tmp.rglob("*.tar.gz")):
        subprocess.run(
            ["unzip", "-o", "-q", str(arch), "-d", str(arch.parent)]
            if arch.suffix == ".zip"
            else ["tar", "xzf", str(arch), "-C", str(arch.parent)],
            check=False,
        )

    # If gdown created a single wrapper dir, unwrap it
    entries = [p for p in tmp.iterdir()]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
    bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, bundle, dirs_exist_ok=True)
    assets_volume.commit()
    return f"GDOWN LOG (tail):\n{log}\n\nBUNDLE TREE:\n" + _tree(bundle)


@app.local_entrypoint()
def download_assets(force: bool = False):
    print(download_assets_remote.remote(force=force))


# ---------------------------------------------------------------------------
# Step 2: unpack SMPL-X body models (uploaded via `modal volume put`)
# ---------------------------------------------------------------------------

@app.function(image=dart_image, volumes={"/vol": assets_volume}, timeout=900)
def setup_body_models_remote() -> str:
    import subprocess

    zip_path = Path("/vol/smplx_lockedhead_20230207.zip")
    if not zip_path.exists():
        return ("BLOCKED: /vol/smplx_lockedhead_20230207.zip not found. "
                "Upload with: modal volume put dart-assets "
                "inbox/smplx_lockedhead_20230207.zip /smplx_lockedhead_20230207.zip")
    dest = Path("/vol/body_models/smplx_lockedhead_20230207")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest)], check=True)
    assets_volume.commit()
    return "BODY MODELS:\n" + _tree(dest)


@app.local_entrypoint()
def setup_body_models():
    print(setup_body_models_remote.remote())


# ---------------------------------------------------------------------------
# Step 3: scene prep -- Y-up GLB -> Z-up OBJ + mesh2sdf grid
# ---------------------------------------------------------------------------

@app.function(image=dart_image, volumes={"/vol": assets_volume}, timeout=1800)
def prep_scene_remote(glb_bytes: bytes, size: int = 256) -> str:
    import mesh2sdf
    import numpy as np
    import trimesh

    mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh")
    # Y-up (glTF) -> Z-up (DART/AMASS): (x, y, z) -> (x, -z, y)
    yup_to_zup = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
    )
    mesh.apply_transform(yup_to_zup)

    scene_dir = Path("/vol/scenes/test_scene")
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(scene_dir / "scene_with_floor.obj")

    # SDF exactly per DART's scenes/test_sdf.py (mesh_scale=0.8, fix=True)
    mesh_scale = 0.8
    vertices = mesh.vertices
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    sdf, _fixed = mesh2sdf.compute(
        (vertices - center) * scale, mesh.faces, size,
        fix=True, level=1.0 / size, return_mesh=True,
    )
    np.save(scene_dir / "scene_sdf.npy", sdf)
    cfg = {"center": center.tolist(), "scale": scale, "level": 1.0 / size, "size": size}
    with open(scene_dir / "scene_sdf.json", "w") as f:
        json.dump(cfg, f)
    assets_volume.commit()

    # Sanity probes (Z-up coords)
    probes = {
        "crate_center (3.5,2.5,0.5) expect<0": [3.5, 2.5, 0.5],
        "free_air (1.0,0.5,1.5)     expect>0": [1.0, 0.5, 1.5],
        "path_midair (2.0,1.3,0.9)  expect>0": [2.0, 1.3, 0.9],
        "below_floor (0,0,-0.5)     expect<0": [0.0, 0.0, -0.5],
        "goal_free (6.0,4.5,0.9)    expect>0": [6.0, 4.5, 0.9],
    }
    vals = _calc_sdf_cpu(sdf, cfg, np.array(list(probes.values())))
    report = [f"scene bbox zup min={bbmin.round(2).tolist()} max={bbmax.round(2).tolist()}",
              f"sdf grid {size}^3, voxel ~{(bbmax - bbmin).max() / mesh_scale / size * 2 * 100:.1f} cm"]
    report += [f"  {k}: {v:+.3f} m" for k, v in zip(probes, vals)]
    return "\n".join(report)


@app.local_entrypoint()
def prep_scene():
    glb = SCENE_GLB.read_bytes()
    print(f"Uploading {SCENE_GLB} ({len(glb) / 1024:.0f} KB)")
    print(prep_scene_remote.remote(glb))


# ---------------------------------------------------------------------------
# Step 4: stock demo -- text-to-motion rollout (proof the model runs)
# ---------------------------------------------------------------------------

@app.function(
    image=dart_image, volumes={"/vol": assets_volume}, gpu="A10G", timeout=3600
)
def stock_rollout_remote(text_prompt: str = "walk in circles*20") -> dict:
    import os
    import subprocess

    _assemble_workspace()
    os.chdir("/DART")
    cmd = [
        "python", "-m", "mld.rollout_mld",
        "--denoiser_checkpoint", CKPT,
        "--text_prompt", text_prompt,
        "--batch_size", "2",
        "--guidance_param", "5.0",
        "--respacing", "",
        "--export_smpl", "1",
        "--zero_noise", "0",
        "--use_predicted_joints", "1",
        "--dataset", "babel",
        "--fix_floor", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/DART")
    log = proc.stdout[-6000:] + "\nSTDERR:\n" + proc.stderr[-6000:]

    files = {}
    ckpt_dir = Path(CKPT).parent / Path(CKPT).stem
    for p in sorted((Path("/DART") / ckpt_dir / "rollout").rglob("*")):
        if p.is_file() and p.suffix in (".npz", ".pkl"):
            files[str(p.relative_to("/DART"))] = p.read_bytes()
    assets_volume.commit()
    return {"log": log, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def stock_rollout(text_prompt: str = "walk in circles*20"):
    res = stock_rollout_remote.remote(text_prompt=text_prompt)
    print(res["log"])
    dest = OUT_DIR / "stock_rollout"
    dest.mkdir(parents=True, exist_ok=True)
    for rel, data in res["files"].items():
        out = dest / Path(rel).name
        out.write_bytes(data)
        print(f"saved {out} ({len(data) / 1024:.0f} KB)")
    print(f"returncode={res['returncode']}")


# ---------------------------------------------------------------------------
# Step 5: THE experiment -- (a) stock waypoint vs (b) + SDF penetration loss
# ---------------------------------------------------------------------------

@app.function(
    image=dart_image, volumes={"/vol": assets_volume}, gpu="A10G", timeout=7200
)
def experiment_remote(
    optim_steps: int = 100,
    batch_size: int = 4,
    text_prompt: str = TEXT_PROMPT,
) -> dict:
    import os
    import pickle
    import subprocess

    import numpy as np
    import torch

    _assemble_workspace()
    os.chdir("/DART")

    # ---- init joints: standing pose at START_XY facing the goal (Z-up) ----
    import smplx
    from pytorch3d import transforms as p3dt

    with open("data/stand.pkl", "rb") as f:
        stand = pickle.load(f)

    bm = smplx.build_layer(
        "data/smplx_lockedhead_20230207/models_lockedhead",
        model_type="smplx", gender="male", ext="npz", num_pca_comps=12,
    )

    def to_rotmat(arr, n):
        t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
        if t.shape[-2:] == (3, 3):
            return t.reshape(1, n, 3, 3)
        return p3dt.axis_angle_to_matrix(t.reshape(n, 3)).reshape(1, n, 3, 3)

    frame = 0
    go = to_rotmat(np.asarray(stand["global_orient"])[frame], 1)
    bp = to_rotmat(np.asarray(stand["body_pose"])[frame], 21)
    transl = torch.as_tensor(
        np.asarray(stand["transl"])[frame], dtype=torch.float32
    ).reshape(1, 3)
    out = bm(betas=torch.zeros(1, 10), global_orient=go, body_pose=bp, transl=transl)
    j = out.joints[0, :22].detach().numpy()  # pelvis=0, l_hip=1, r_hip=2 (Z-up)
    feet_z = float(min(j[7][2], j[8][2], j[10][2], j[11][2]))  # ankles+feet
    pelvis, lhip, rhip = j[0].copy(), j[1].copy(), j[2].copy()

    # current facing: y = z x (rhip - lhip), z-component zeroed
    d = rhip - lhip
    cur = np.array([-d[1], d[0], 0.0])
    cur /= np.linalg.norm(cur)
    goal_dir = np.array([GOAL_XY[0] - START_XY[0], GOAL_XY[1] - START_XY[1], 0.0])
    goal_dir /= np.linalg.norm(goal_dir)
    ang = np.arctan2(goal_dir[1], goal_dir[0]) - np.arctan2(cur[1], cur[0])
    R = np.array([
        [np.cos(ang), -np.sin(ang), 0.0],
        [np.sin(ang), np.cos(ang), 0.0],
        [0.0, 0.0, 1.0],
    ])
    new_pelvis = np.array([START_XY[0], START_XY[1], pelvis[2] - feet_z])
    lift = new_pelvis - pelvis  # translation after rotating about pelvis
    init_joints = [
        new_pelvis.tolist(),
        (R @ (lhip - pelvis) + new_pelvis).tolist(),
        (R @ (rhip - pelvis) + new_pelvis).tolist(),
    ]
    goal_joints = [[GOAL_XY[0], GOAL_XY[1], float(new_pelvis[2])]]
    geom_note = (
        f"stand pelvis z={pelvis[2]:.3f}, feet z={feet_z:.3f} -> "
        f"init pelvis z={new_pelvis[2]:.3f}; facing {goal_dir.round(3).tolist()}"
    )

    # ---- write interaction configs (identical except name) ----
    runs = {
        "a_stock": {"weight_collision": 0.0, "weight_contact": 0.0},
        "b_sdf": {"weight_collision": 0.1, "weight_contact": 0.1},
    }
    logs, results = {}, {}
    for tag, w in runs.items():
        cfg = {
            "scene_dir": "/vol/scenes/test_scene",
            "floor_height": 0.0,
            "interaction_name": f"crate_{tag}",
            "interactions": [{
                "text_prompt": text_prompt,
                "init_joints": init_joints,
                "goal_joints": goal_joints,
            }],
        }
        cfg_path = Path(f"/DART/data/optim_interaction/crate_{tag}.json")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=1))

        cmd = [
            "python", "-m", "mld.optim_scene_mld",
            "--denoiser_checkpoint", CKPT,
            "--interaction_cfg", str(cfg_path),
            "--optim_lr", "0.01",
            "--optim_steps", str(optim_steps),
            "--batch_size", str(batch_size),
            "--guidance_param", "5.0",
            "--respacing", "ddim10",
            "--export_smpl", "1",
            "--use_predicted_joints", "1",
            "--optim_unit_grad", "1",
            "--optim_anneal_lr", "1",
            "--weight_jerk", "0.1",
            "--weight_collision", str(w["weight_collision"]),
            "--weight_contact", str(w["weight_contact"]),
            "--weight_skate", "0.0",
            "--contact_thresh", "0.00",
            "--load_cache", "0",
            "--init_noise_scale", "0.1",
            "--seed", "0",
        ]
        print(f"=== RUN {tag}: collision={w['weight_collision']} "
              f"contact={w['weight_contact']} ===")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/DART")
        # keep loss head+tail: shows optimization trajectory
        head = "\n".join(proc.stdout.splitlines()[:30])
        tail = "\n".join(proc.stdout.splitlines()[-30:])
        logs[tag] = (f"rc={proc.returncode}\n{head}\n...\n{tail}\n"
                     f"STDERR tail:\n{proc.stderr[-3000:]}")
        if proc.returncode != 0:
            continue
        optim_dir = Path("/DART") / Path(CKPT).parent / Path(CKPT).stem / "optim"
        hits = sorted(optim_dir.glob(f"crate_{tag}_*"))
        if hits:
            results[tag] = hits[-1]

    # ---- analysis + evidence ----
    import matplotlib.pyplot as plt

    sdf_grid = np.load("/vol/scenes/test_scene/scene_sdf.npy")
    sdf_cfg = json.loads(Path("/vol/scenes/test_scene/scene_sdf.json").read_text())
    (cx, cy, cz) = CRATE_BOX

    metrics = {"geometry_note": geom_note, "start": START_XY, "goal": GOAL_XY,
               "text_prompt": text_prompt, "optim_steps": optim_steps,
               "runs": {}}
    files = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    colors = {"a_stock": "crimson", "b_sdf": "seagreen"}

    for tag, run_dir in results.items():
        run_metrics = {"samples": {}}
        for pkl_path in sorted(run_dir.glob("sample_*.pkl")):
            if "smplx" in pkl_path.name:
                continue
            with open(pkl_path, "rb") as f:
                seq = pickle.load(f)
            joints = np.asarray(seq["joints"])  # [T, 22, 3] Z-up world
            T = joints.shape[0]
            inside = (
                (joints[..., 0] > cx[0]) & (joints[..., 0] < cx[1])
                & (joints[..., 1] > cy[0]) & (joints[..., 1] < cy[1])
                & (joints[..., 2] > cz[0]) & (joints[..., 2] < cz[1])
            )
            all_sdf = _calc_sdf_cpu(sdf_grid, sdf_cfg, joints.reshape(-1, 3))
            goal_err = float(np.linalg.norm(
                joints[-1, 0, :2] - np.array(GOAL_XY)))
            idx = pkl_path.stem.split("_")[-1]
            run_metrics["samples"][idx] = {
                "frames": int(T),
                "frames_any_joint_in_crate": int(inside.any(axis=1).sum()),
                "pct_frames_in_crate": round(
                    100.0 * inside.any(axis=1).sum() / T, 1),
                "min_joint_sdf_m": round(float(all_sdf.min()), 3),
                "final_pelvis_goal_err_m": round(goal_err, 3),
            }
            pel = joints[:, 0]
            axes[0].plot(pel[:, 0], pel[:, 1], color=colors[tag], alpha=0.7,
                         lw=1.5, label=f"{tag} s{idx}" if idx == "0" else None)
            axes[1].plot(np.linspace(0, 1, T), pel[:, 2], color=colors[tag],
                         alpha=0.7, lw=1.2)
            files[f"experiment/{tag}/{pkl_path.name}"] = pkl_path.read_bytes()
        for npz_path in sorted(run_dir.glob("sample_*_smplx.npz")):
            files[f"experiment/{tag}/{npz_path.name}"] = npz_path.read_bytes()
        metrics["runs"][tag] = run_metrics

    # scene footprint (Z-up)
    ax = axes[0]
    for (bx, by, label) in [
        ((3.0, 4.0), (2.0, 3.0), "big crate"),
        ((-3.9, -3.3), (-2.5, -1.9), "small crate"),
        ((-2.5, 2.5), (5.85, 6.15), "back wall"),
        ((6.35, 6.65), (-3.0, 1.0), "side wall"),
        ((-4.0, -1.6), (2.8, 4.0), "platform"),
        ((2.0, 4.4), (-4.0, -2.6), "ramp"),
    ]:
        ax.add_patch(plt.Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                                   fc="tan", ec="saddlebrown", alpha=0.6))
        ax.annotate(label, ((bx[0] + bx[1]) / 2, (by[0] + by[1]) / 2),
                    ha="center", fontsize=7)
    ax.plot(*zip(START_XY, GOAL_XY), "k--", lw=1, label="straight line")
    ax.plot(*START_XY, "k^", ms=10)
    ax.plot(*GOAL_XY, "k*", ms=14)
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_aspect("equal")
    ax.set_title("Top-down (Z-up world): pelvis paths\n"
                 "red = (a) stock waypoint, green = (b) + SDF loss")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="lower left", fontsize=7)
    axes[1].axhspan(0.0, 1.0, xmin=0.45, xmax=0.55, color="tan", alpha=0.4)
    axes[1].set_title("pelvis height vs normalized time (crate zone shaded)")
    axes[1].set_xlabel("t (0=start, 1=goal)"); axes[1].set_ylabel("z [m]")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    files["experiment/evidence_topdown.png"] = buf.getvalue()

    assets_volume.commit()
    return {"logs": logs, "metrics": metrics, "files": files}


@app.local_entrypoint()
def experiment(optim_steps: int = 100, batch_size: int = 4,
               text_prompt: str = TEXT_PROMPT):
    res = experiment_remote.remote(
        optim_steps=optim_steps, batch_size=batch_size, text_prompt=text_prompt)
    for tag, log in res["logs"].items():
        print(f"\n===== {tag} =====\n{log}")
    print("\n===== METRICS =====")
    print(json.dumps(res["metrics"], indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "experiment_metrics.json").write_text(
        json.dumps(res["metrics"], indent=2))
    for rel, data in res["files"].items():
        out = OUT_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"saved {out} ({len(data) / 1024:.0f} KB)")
