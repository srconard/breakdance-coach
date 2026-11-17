"""Render a track_record.py recording: reference (left) | physics (right).

Also writes metrics: per-step mean body-position error (all robot bodies),
the share of steps under 0.15 m / 0.5 m, and whether/when the body fell
(pelvis more than 0.5 m from the reference pelvis).
    MUJOCO_GL=osmesa python render_compare.py rec.npz out.mp4 [--size 480]
"""
import argparse
import json
import subprocess

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    r = np.load(a.rec)
    model = mujoco.MjModel.from_binary_path(a.rec[:-4] + ".mjb")
    Q, RQ = r["qpos"], r["ref_qpos"]
    n_rq = RQ.shape[1]
    if not bool(r["w_last"]):
        pass  # convert_to_sim already gave wxyz
    else:  # sim used xyzw: MuJoCo qpos is wxyz regardless
        RQ[:, 3:7] = RQ[:, [6, 3, 4, 5]]
    ds, dr = mujoco.MjData(model), mujoco.MjData(model)
    # robot bodies = bodies whose root ancestor is the free-joint tree of qpos[0:7]
    root_body = model.jnt_bodyid[0]
    robot = [b for b in range(model.nbody) if model.body_rootid[b] == root_body]
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.6
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.size)
    ren = mujoco.Renderer(model, a.size, a.size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.2, -22.0, 135.0
    robot_geoms = np.isin(model.geom_bodyid, robot)
    rgba0 = model.geom_rgba.copy()
    fps = 1.0 / float(r["dt"])
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{2 * a.size}x{a.size}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow",
         a.out], stdin=subprocess.PIPE)
    # the sim respawns each episode at a terrain offset: align xy per episode
    done = np.array(r["done"]).astype(bool)
    starts = [0] + [i + 1 for i in np.nonzero(done)[0] if i + 1 < len(Q)]
    off = np.zeros((len(Q), 2))
    for k, s0 in enumerate(starts):
        s1 = starts[k + 1] if k + 1 < len(starts) else len(Q)
        off[s0:s1] = Q[s0, :2] - RQ[s0, :2]
    err, fell_at = [], None
    look_s = look_r = None
    for i in range(len(Q)):
        ds.qpos[:] = Q[i]
        ds.qpos[:2] -= off[i]
        dr.qpos[:] = ds.qpos
        dr.qpos[:n_rq] = RQ[i]
        mujoco.mj_forward(model, ds)
        mujoco.mj_forward(model, dr)
        e = np.linalg.norm(ds.xpos[robot] - dr.xpos[robot], axis=1).mean()
        err.append(float(e))
        if fell_at is None and np.linalg.norm(ds.xpos[root_body] - dr.xpos[root_body]) > 0.5:
            fell_at = i
        if i in starts:
            look_s = look_r = None
        look_s = ds.xpos[root_body].copy() if look_s is None else 0.85 * look_s + 0.15 * ds.xpos[root_body]
        look_r = dr.xpos[root_body].copy() if look_r is None else 0.85 * look_r + 0.15 * dr.xpos[root_body]
        frames = []
        for data, col, look in ((dr, (0.35, 0.55, 0.95, 1.0), look_r), (ds, (0.95, 0.55, 0.25, 1.0), look_s)):
            model.geom_rgba[robot_geoms] = col
            cam.lookat[:] = [look[0], look[1], 0.45]
            ren.update_scene(data, camera=cam)
            frames.append(ren.render())
        model.geom_rgba[:] = rgba0
        ff.stdin.write(np.hstack(frames).tobytes())
    ff.stdin.close()
    ff.wait()
    err = np.array(err)
    m = {"steps": len(err), "seconds": round(len(err) / fps, 2),
         "mean_body_err_m": round(float(err.mean()), 3),
         "max_body_err_m": round(float(err.max()), 3),
         "pct_steps_err_lt_0.15m": round(100 * float((err < 0.15).mean()), 1),
         "pct_steps_err_lt_0.5m": round(100 * float((err < 0.5).mean()), 1),
         "fell_at_s": None if fell_at is None else round(fell_at / fps, 2),
         "resets": int(done.sum()),
         "pelvis_z_sim_mean": round(float(np.mean(Q[:, 2])), 3),
         "pelvis_z_ref_mean": round(float(np.mean(RQ[:, 2])), 3)}
    # per-episode error, so a failing first pass and a later pass read apart
    m["per_episode_mean_err_m"] = [round(float(err[a:b].mean()), 3) for a, b in
                                   zip(starts, starts[1:] + [len(err)]) if b > a]
    print("[render_compare]", json.dumps(m))
    with open(a.out[:-4] + "_metrics.json", "w") as f:
        json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
