"""SO-101 PICK environment: grasp the box and lift it 5 cm. Builds on the reach env.

Lessons carried over from reach (notes/02-06):
  - success does NOT end the episode (reward is positive, ending early would cost reward)
  - nothing terminates early: knocking/pushing the box is allowed while learning
  - the obs contains the motor targets (Markov) and is normalised by VecNormalize at train time
New for pick (notes/07):
  - the gripper's ORIENTATION matters (fingers down, jaws either side), so it goes in the obs
  - jaw-box contact flags go in the obs: "am I touching it with each jaw?"

observation (41): qpos 6 | qvel 6 | claw xyz 3 | box xyz 3 | grasp_point - claw 3 | ctrl 6
                  | finger axis 3 | jaw-closing axis 3 | box velocity 3 | box up-axis 3
                  | fixed-jaw touching box 1 | moving-jaw touching box 1
action (6)      : delta joint targets incl. the gripper, same as reach
reward          : staged, each stage only pays once the previous is reached
    r = reach                       1 - tanh(5 * |grasp_point - claw|)          in [0, 1]
      + 0.5 * down                  how much the fingers point down             in [0, 0.5]
      + grasped * (0.5 + 6 * lift)  both jaws on AND squeezing, then height     in [0, 6.5]
  v1 paid grasped*(1 + 3*lift) for mere contact -> policy rested its jaws on the box and sat (notes/08)
  'grasped *' blocks the cheat of earning lift reward by flicking/pushing the box up.
success         : grasped AND box lifted >= 5 cm
"""
import mujoco
import numpy as np
import gymnasium as gym

from envs.so101_reach import BOX_Z, FRAME_SKIP, MAX_DELTA, MAX_LEAD, SO101ReachEnv

MAX_STEPS = 150          # 7.5 s: reach + align + close + lift needs longer than reach
GRASP_ABOVE_CENTRE = 0.01
LIFT_FULL = 0.08         # lift reward saturates at 8 cm
SUCCESS_LIFT = 0.05
SQUEEZE_MIN = 0.05       # rad: v1 counted jaws merely RESTING on the box as a grasp (notes/08)


class SO101PickEnv(SO101ReachEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode, terminate_on_knock=False)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(41,), dtype=np.float64)
        m = self.model
        self.site_id = m.site("gripperframe").id
        self.box_geom = m.geom("box").id
        self.fixed_jaw_body = m.body("gripper").id
        self.moving_jaw_body = m.body("moving_jaw_so101_v1").id
        self.box_dof_adr = m.jnt_dofadr[m.body("box").jntadr[0]]

    def grasp_point(self):
        return self.box_pos() + np.array([0.0, 0.0, GRASP_ABOVE_CENTRE])

    def jaw_contacts(self):
        fixed = moving = False
        d, m = self.data, self.model
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if self.box_geom not in (g1, g2):
                continue
            other = m.geom_bodyid[g2 if g1 == self.box_geom else g1]
            fixed |= other == self.fixed_jaw_body
            moving |= other == self.moving_jaw_body
        return fixed, moving

    def _get_obs(self):
        d = self.data
        R = d.site_xmat[self.site_id].reshape(3, 3)
        box_R = d.body("box").xmat.reshape(3, 3)
        fixed, moving = self.jaw_contacts()
        claw = self.claw_pos()
        return np.concatenate([
            d.qpos[:6], d.qvel[:6], claw, self.box_pos(), self.grasp_point() - claw, d.ctrl,
            R[:, 0],                        # finger axis (want: pointing down)
            R[:, 2],                        # jaw-closing axis
            d.qvel[self.box_dof_adr:self.box_dof_adr + 3],   # box linear velocity
            box_R[:, 2],                    # box up-axis: is it tipped over?
            [float(fixed), float(moving)],
        ])

    def reward_info(self):
        """Reward + diagnostics for the CURRENT state (also used by the scripted oracle)."""
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        dist = float(np.linalg.norm(self.grasp_point() - self.claw_pos()))
        reach = 1.0 - np.tanh(5.0 * dist)
        down = max(0.0, -R[2, 0])                      # finger axis . (0,0,-1)
        fixed, moving = self.jaw_contacts()
        squeeze = self.data.qpos[5] - self.data.ctrl[5]   # >0: motor target more closed than the jaw -> force on the box
        grasped = fixed and moving and squeeze > SQUEEZE_MIN
        height = self.box_pos()[2] - BOX_Z
        lift = float(np.clip(height / LIFT_FULL, 0.0, 1.0))
        reward = reach + 0.5 * down + grasped * (0.5 + 6.0 * lift)
        info = {"distance": dist, "down": down, "grasped": grasped, "height": height,
                "is_success": bool(grasped and height >= SUCCESS_LIFT), "knocked": False}
        return float(reward), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = self.data.ctrl + MAX_DELTA * action
        q = self.data.qpos[:6]
        target = np.clip(target, q - MAX_LEAD, q + MAX_LEAD)
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        reward, info = self.reward_info()
        truncated = self.steps >= MAX_STEPS
        return self._get_obs(), reward, False, bool(truncated), info
