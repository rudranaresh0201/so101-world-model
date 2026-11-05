"""SO-101 REACH environment: the claw must touch the box. YOU fill in the TODOs.

Everything we designed (PLAN.md / chat, "lock-in check"):
  observation : 6 joint angles + 6 joint speeds + claw xyz + box xyz + (box - claw)  = 21 numbers
  action      : 6 numbers in [-1, 1] = DELTA joint targets, at most +-0.05 rad per step
  control     : one decision every 0.05 s = hold each action for 10 physics steps
  reward      : 1 - tanh(5 * distance)        (always >= 0, so ending early never pays)
  terminated  : success (distance < 0.02 m) OR box knocked > 0.05 m sideways from where it started
  truncated   : 100 decisions reached (time limit)
  reset       : arm at home, box at a RANDOM spot within reach: x in [0.15, 0.30], y in [-0.12, 0.12]

Test it with:   python scripts/check_env.py
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "third_party/mujoco_menagerie/robotstudio_so101/scene_box.xml"

MAX_DELTA = 0.05        # rad: biggest joint-target change one action can make
FRAME_SKIP = 10         # physics steps per decision (10 x 0.005 s = 0.05 s)
MAX_STEPS = 100         # decisions per episode (= 5 seconds)
SUCCESS_DIST = 0.02     # m
KNOCK_DIST = 0.05       # m
BOX_X_RANGE = (0.15, 0.30)
BOX_Y_RANGE = (-0.12, 0.12)
BOX_Z = 0.03            # the box is 6 cm tall, so its centre rests at z = 0.03


class SO101ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None):
        # ---- given: load the robot, nothing to do here ----
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()   # per-joint min target (rad)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()  # per-joint max target (rad)
        # where the box's 7 numbers (xyz + quaternion) live inside data.qpos
        self.box_qpos_adr = self.model.joint(self.model.body("box").jntadr[0]).qposadr[0]

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)

        self.steps = 0
        self.box_start = np.zeros(3)

    # ---- given: helpers ----
    def claw_pos(self) -> np.ndarray:
        return self.data.site("gripperframe").xpos.copy()

    def box_pos(self) -> np.ndarray:
        return self.data.body("box").xpos.copy()

    # ------------------------------------------------------------------ TODO 1
    def _get_obs(self) -> np.ndarray:
        """Return the 21-number observation, in this order:
        joint angles (data.qpos[:6]), joint speeds (data.qvel[:6]),
        claw xyz, box xyz, box - claw.
        Hint: np.concatenate([...])"""
        raise NotImplementedError("TODO 1: build the observation")

    # ------------------------------------------------------------------ TODO 2
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)          # sets up self.np_random -- ALWAYS use it for randomness
        mujoco.mj_resetData(self.model, self.data)   # arm back to home (all joints 0)

        # TODO 2a: put the box at a random (x, y) inside BOX_X_RANGE / BOX_Y_RANGE, at height BOX_Z.
        #          Write into self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3]
        #          Use self.np_random.uniform(low, high)  (NOT np.random -- seeds must work)
        raise NotImplementedError("TODO 2a: randomise the box position")

        self.data.ctrl[:] = 0.0                      # motor targets = home pose
        mujoco.mj_forward(self.model, self.data)     # recompute positions after we moved things

        # TODO 2b: remember where the box started (self.box_start) and reset the step counter
        raise NotImplementedError("TODO 2b: box_start + step counter")

        return self._get_obs(), {}

    # ------------------------------------------------------------------ TODO 3
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # TODO 3a: DELTA action. New target = current target (self.data.ctrl) + MAX_DELTA * action,
        #          clipped to [self.ctrl_low, self.ctrl_high]. Write it back into self.data.ctrl.
        raise NotImplementedError("TODO 3a: apply the delta action")

        # TODO 3b: run FRAME_SKIP physics steps:  mujoco.mj_step(self.model, self.data)
        raise NotImplementedError("TODO 3b: simulate")

        self.steps += 1
        distance = float(np.linalg.norm(self.claw_pos() - self.box_pos()))

        # TODO 3c: reward = 1 - tanh(5 * distance)
        reward = None

        # TODO 3d: the two endings
        #   success = distance < SUCCESS_DIST
        #   knocked = the box moved more than KNOCK_DIST *sideways* (x, y only) from self.box_start
        #   terminated = success or knocked
        #   truncated  = time limit reached
        success = knocked = terminated = truncated = None

        info = {"distance": distance, "is_success": success, "knocked": knocked}
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ---- given: rendering for videos ----
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.2, 0.0, 0.1]
        cam.distance, cam.azimuth, cam.elevation = 0.8, 140, -30
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
