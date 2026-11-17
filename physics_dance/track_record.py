"""Run a ProtoMotions tracker headless and record sim + reference states.

Wraps protomotions/inference_agent.py: the evaluator's interactive
simple_test_policy loop is replaced by a fixed-length recording loop. Runs in
the ProtoMotions checkout (cwd = repo root). Environment:
    PD_OUT    output .npz (qpos, ref_qpos, t, dt, done) + .mjb model beside it
    PD_STEPS  number of control steps to record
Usage (same args as inference_agent.py):
    PD_OUT=/tmp/x.npz PD_STEPS=200 python track_record.py --simulator mujoco \
        --num-envs 1 --headless --checkpoint ... --motion-file ...
"""
import os
import sys

import numpy as np

OUT = os.environ["PD_OUT"]
NSTEPS = int(os.environ["PD_STEPS"])


def record(self):
    import mujoco
    import torch

    self.agent.eval()
    env = self.env
    sim = env.simulator
    ref = env.reference_motion
    mm, ml = ref.motion_manager, ref.motion_lib
    Q, RQ, T, D = [], [], [], []
    done_indices = None
    with torch.no_grad():
        for step in range(NSTEPS):
            obs, _ = env.reset(done_indices)
            self.agent.pre_collect_step(step)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)
            action = self._policy_action(obs_td)
            _, _, dones, _, _ = env.step(action)
            ids, t = mm.motion_ids.clone(), mm.motion_times.clone()
            st = ml.get_motion_state(ids, t).convert_to_sim(sim.data_conversion)
            rq = np.concatenate([
                st.rigid_body_pos[0, 0].cpu().numpy(),
                st.rigid_body_rot[0, 0].cpu().numpy(),
                st.dof_pos[0].cpu().numpy()])
            Q.append(np.array(sim.data.qpos, copy=True))
            RQ.append(rq)
            T.append(float(t[0]))
            D.append(bool(dones[0]))
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
    np.savez(OUT, qpos=np.array(Q), ref_qpos=np.array(RQ), t=np.array(T),
             done=np.array(D), dt=float(mm.env_dt),
             w_last=bool(sim.data_conversion.sim_w_last))
    mujoco.mj_saveModel(sim.model, OUT[:-4] + ".mjb", None)
    print(f"[track_record] {len(Q)} steps -> {OUT}")


if __name__ == "__main__":
    from protomotions.agents.evaluators import base_evaluator

    base_evaluator.BaseEvaluator.simple_test_policy = record
    from protomotions.inference_agent import main

    main(sys.argv[1:])
