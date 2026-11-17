"""Warm-start fine-tune of NVIDIA's anatomical-SOMA tracker on ONE clip,
on a single Modal GPU with ProtoMotions' Newton backend (MuJoCo-Warp physics;
no Isaac Sim, no account). The clip must already be in the `physics-dance`
volume as /vol/<clip>/anat.pt (modal_soma_track.py writes it).

    modal run physics_dance/modal_finetune.py --clip sixstep_fw1 --minutes 25

Writes /vol/<clip>/ft/<run>/ (last.ckpt, inference checkpoint, configs) and
returns the training log tail. Evaluate in MuJoCo afterwards with
    modal run physics_dance/modal_soma_track.py --name sixstep_fw1@ft \
        --checkpoint /vol/sixstep_fw1/ft/<run>/last.ckpt
Hard wall-clock cap = --minutes (the subprocess is killed), so the spend is
bounded: a T4 is ~$0.59/h (L4+ need a payment method on the workspace).
"""
import modal

PM_COMMIT = "13de0cc"
CKPT = "data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq/inference_last.ckpt"
EXP = "data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq/experiment_config.py"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "ffmpeg", "libosmesa6", "libosmesa6-dev", "libgl1",
                 "libgl1-mesa-dev", "libglu1-mesa-dev", "libglib2.0-0", "build-essential",
                 "cmake", "libx11-dev", "libxrandr-dev", "libxinerama-dev", "libxcursor-dev",
                 "libxi-dev", "xvfb", "patchelf")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("newton[examples]==1.0.0")
    .run_commands(
        "git lfs install",
        "GIT_LFS_SKIP_SMUDGE=1 git -c core.hooksPath=/dev/null clone https://github.com/NVlabs/ProtoMotions.git /opt/pm",
        f"cd /opt/pm && git -c core.hooksPath=/dev/null checkout {PM_COMMIT}",
        "cd /opt/pm && git lfs pull --include='data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq/*'",
        "cd /opt/pm && pip install -r requirements_newton.txt && pip install -e .",
    )
)
app = modal.App("physics-dance-finetune", image=image)
vol = modal.Volume.from_name("physics-dance", create_if_missing=True)


@app.function(gpu="T4", cpu=4.0, memory=24576, timeout=3600, volumes={"/vol": vol})
def finetune(clip: str, minutes: float, num_envs: int = 2048, run: str = "ft",
             extra: str = "") -> dict:
    import os
    import subprocess
    import time

    os.makedirs(f"/vol/{clip}/{run}", exist_ok=True)
    os.makedirs("/opt/pm/results", exist_ok=True)
    if not os.path.islink(f"/opt/pm/results/{clip}_{run}"):
        os.symlink(f"/vol/{clip}/{run}", f"/opt/pm/results/{clip}_{run}")
    cmd = (f"python protomotions/train_agent.py --robot-name anatomic_soma --simulator newton "
           f"--experiment-path {EXP} --experiment-name {clip}_{run} "
           f"--motion-file /vol/{clip}/anat.pt --checkpoint {CKPT} "
           f"--num-envs {num_envs} --batch-size {num_envs * 4} --headless {extra}")
    t0 = time.time()
    logf = open(f"/vol/{clip}/{run}/train.log", "a")
    p = subprocess.Popen(cmd, shell=True, cwd="/opt/pm", stdout=logf, stderr=subprocess.STDOUT)
    rc = None
    while time.time() - t0 < minutes * 60:
        rc = p.poll()
        if rc is not None:
            break
        time.sleep(10)
        if int(time.time() - t0) % 120 < 10:
            vol.commit()
    if rc is None:
        p.terminate()
        try:
            p.wait(60)
        except Exception:
            p.kill()
    logf.close()
    vol.commit()
    log = open(f"/vol/{clip}/{run}/train.log").read()
    files = sorted(os.listdir(f"/vol/{clip}/{run}"))
    return {"rc": rc, "seconds": round(time.time() - t0), "files": files, "log": log[-8000:]}


@app.local_entrypoint()
def main(clip: str, minutes: float = 3.0, num_envs: int = 2048, run: str = "ft", extra: str = ""):
    res = finetune.remote(clip, minutes, num_envs, run, extra)
    print("RESULT", {k: res[k] for k in ("rc", "seconds", "files")})
    print(res["log"][-6000:])
