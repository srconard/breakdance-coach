"""Six-step (or any baked clip) -> pretrained physics body, on Modal CPU.

Pipeline (all inside one Modal CPU container, no GPU):
  bake npz -> SOMA23 .motion (soma_from_bake.py) -> packaged MotionLib
  -> anatomical-SOMA retarget (ProtoMotions' PyRoki/JAX solver)
  -> NVIDIA's released anatomical-SOMA BONES-SEED FSQ tracker in MuJoCo
     (model card: trained in IsaacLab, verified in MuJoCo; NVIDIA Open Model
     License; personal research use here)
  -> side-by-side video (reference | physics) + tracking metrics.

Run from the repo root on a machine with a Modal token:
    modal run physics_dance/modal_soma_track.py --npz output/physics_dance/bake/mixamo_fw1.npz --name sixstep_fw1
"""
import json
from pathlib import Path

import modal

PM_COMMIT = "13de0cc"  # ProtoMotions3 head on 2026-11-17
CKPT = "data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq/inference_last.ckpt"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "ffmpeg", "libosmesa6", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "git lfs install",
        "GIT_LFS_SKIP_SMUDGE=1 git -c core.hooksPath=/dev/null clone https://github.com/NVlabs/ProtoMotions.git /opt/pm",
        f"cd /opt/pm && git -c core.hooksPath=/dev/null checkout {PM_COMMIT}",
        "cd /opt/pm && git lfs pull --include='data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq/*'",
        "cd /opt/pm && pip install -e . && pip install -r requirements_mujoco.txt",
        "python -m venv /opt/pyroki && /opt/pyroki/bin/pip install -q 'jax[cpu]' jaxlie yourdfpy scipy numpy pytest "
        "'pyroki @ git+https://github.com/chungmin99/pyroki.git'",
    )
    .run_commands(  # NVIDIA's shipped 10-clip anatomical test pack: the control
        "cd /opt/pm && git lfs pull --include='data/motion_for_trackers/anatomic_soma_bones_seed_mini.pt'")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    .add_local_dir(Path(__file__).parent, "/opt/pd", ignore=["__pycache__"])
)

app = modal.App("physics-dance-soma", image=image)
vol = modal.Volume.from_name("physics-dance", create_if_missing=True)


def _sh(cmd, log, env=None, check=True):
    import os
    import subprocess

    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(cmd, shell=True, cwd="/opt/pm", env=e, capture_output=True, text=True)
    log.append(f"$ {cmd}\n[rc {p.returncode}]\n{p.stdout[-6000:]}\n{p.stderr[-6000:]}")
    if check and p.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")
    return p


@app.function(cpu=8.0, memory=16384, timeout=3600, volumes={"/vol": vol})
def track(npz: bytes, name: str, start: float = None, end: float = None, loops: int = 1,
          extra_steps: int = 30, budget_exceed: bool = False, speed: float = 1.0,
          control_steps: int = 0, checkpoint: str = "", keep: bool = True,
          lift: float = 0.0) -> dict:
    import glob
    import os

    import numpy as np

    w = "/tmp/w"
    os.makedirs(f"{w}/motions", exist_ok=True)
    open(f"{w}/in.npz", "wb").write(npz)
    log, out = [], {"name": name}
    ckpt = checkpoint or CKPT
    try:
      if checkpoint and npz == b"":  # evaluate a fine-tuned body on the saved clip
        import shutil
        shutil.copy(f"/vol/{name.split('@')[0]}/anat.pt", f"{w}/anat.pt")
        steps = int(json.load(open(f"/vol/{name.split('@')[0]}/clip.json"))["frames"]) * loops + extra_steps
        _sh(f"python /opt/pd/track_record.py --simulator mujoco --num-envs 1 --headless "
            f"--checkpoint {ckpt} --motion-file {w}/anat.pt",
            log, env={"PD_OUT": f"{w}/rec.npz", "PD_STEPS": str(steps)})
      elif control_steps:  # control: the shipped mini pack, no conversion of ours
        _sh(f"python /opt/pd/track_record.py --simulator mujoco --num-envs 1 --headless "
            f"--checkpoint {CKPT} --motion-file data/motion_for_trackers/anatomic_soma_bones_seed_mini.pt",
            log, env={"PD_OUT": f"{w}/rec.npz", "PD_STEPS": str(control_steps)})
      else:
        se = (f" --start {start}" if start is not None else "") + (f" --end {end}" if end is not None else "")
        _sh(f"python /opt/pd/soma_from_bake.py {w}/in.npz {w}/motions/{name}.motion{se} --loops {loops} --speed {speed} --lift {lift}", log)
        _sh(f"python protomotions/components/motion_lib.py --motion-path {w}/motions/ --output-file {w}/soma.pt", log)
        _sh(f"python data/scripts/from_soma_to_anatomical_soma/retarget_motions.py --input {w}/soma.pt "
            f"--output {w}/anat.pt --pyroki-python /opt/pyroki/bin/python --cache-dir /tmp/jax"
            + (" --allow-motion-budget-exceedance" if budget_exceed else ""), log)
        frames = int(np.load(f"{w}/in.npz")["JW"].shape[0])
        if end is not None or start is not None:
            frames = int(round(((end or frames / 30) - (start or 0)) * 30))
        steps = int(frames / speed) * loops + extra_steps
        if keep:  # keep the anatomical clip for fine-tuning / re-evaluation
            import shutil
            os.makedirs(f"/vol/{name}", exist_ok=True)
            shutil.copy(f"{w}/anat.pt", f"/vol/{name}/anat.pt")
            json.dump({"frames": int(frames / speed) * loops}, open(f"/vol/{name}/clip.json", "w"))
            vol.commit()
        _sh(f"python /opt/pd/track_record.py --simulator mujoco --num-envs 1 --headless "
            f"--checkpoint {ckpt} --motion-file {w}/anat.pt",
            log, env={"PD_OUT": f"{w}/rec.npz", "PD_STEPS": str(steps)})
      _sh(f"python /opt/pd/render_compare.py {w}/rec.npz {w}/{name}.mp4", log)
      out["mp4"] = open(f"{w}/{name}.mp4", "rb").read()
      out["metrics"] = json.load(open(f"{w}/{name}_metrics.json"))
      out["rec"] = open(f"{w}/rec.npz", "rb").read()
      out["mjb"] = open(f"{w}/rec.mjb", "rb").read()
    except Exception as ex:  # return the logs either way
        out["error"] = repr(ex)
    for f in glob.glob(f"{w}/*.json") + glob.glob(f"{w}/*audit*"):
        try:
            out.setdefault("files", {})[os.path.basename(f)] = open(f).read()[-20000:]
        except Exception:
            pass
    out["log"] = "\n".join(log)
    return out


@app.function(cpu=4.0, memory=8192, timeout=1200)
def render_rec(rec: bytes, mjb: bytes, name: str) -> dict:
    import os
    import subprocess

    os.makedirs("/tmp/r", exist_ok=True)
    open("/tmp/r/rec.npz", "wb").write(rec)
    open("/tmp/r/rec.mjb", "wb").write(mjb)
    p = subprocess.run(f"python /opt/pd/render_compare.py /tmp/r/rec.npz /tmp/r/{name}.mp4",
                       shell=True, capture_output=True, text=True)
    out = {"log": p.stdout[-3000:] + p.stderr[-3000:]}
    if p.returncode == 0:
      out["mp4"] = open(f"/tmp/r/{name}.mp4", "rb").read()
      out["metrics"] = json.load(open(f"/tmp/r/{name}_metrics.json"))
    return out


@app.local_entrypoint()
def main(npz: str = "", name: str = "", start: float = None, end: float = None, loops: int = 1,
         budget_exceed: bool = False, rerender: str = "", speed: float = 1.0,
         control_steps: int = 0, checkpoint: str = "", lift: float = 0.0):
    if rerender:  # re-render a saved run dir: --rerender output/physics_dance/runs/<name>
        d = Path(rerender)
        res = render_rec.remote((d / "rec.npz").read_bytes(), (d / "rec.mjb").read_bytes(), d.name)
        if "mp4" in res:
            (d / f"{d.name}.mp4").write_bytes(res["mp4"])
            (d / "metrics.json").write_text(json.dumps(res["metrics"], indent=1))
        print("RESULT", json.dumps(res.get("metrics")), res["log"][-1500:])
        return
    data = Path(npz).read_bytes() if npz else b""
    res = track.remote(data, name, start, end, loops, 30, budget_exceed, speed, control_steps, checkpoint, True, lift)
    od = Path("output/physics_dance/runs") / name
    od.mkdir(parents=True, exist_ok=True)
    (od / "log.txt").write_text(res.get("log", ""), encoding="utf-8")
    for k, v in (res.get("files") or {}).items():
        (od / k).write_text(v, encoding="utf-8")
    if "mp4" in res:
        (od / f"{name}.mp4").write_bytes(res["mp4"])
        (od / "rec.npz").write_bytes(res["rec"])
        (od / "rec.mjb").write_bytes(res["mjb"])
        (od / "metrics.json").write_text(json.dumps(res["metrics"], indent=1))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k in ("name", "metrics", "error")}))
    print(res.get("log", "")[-4000:])
