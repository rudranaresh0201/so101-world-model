"""Pick reward computed from the NEXT observation alone (no simulator needed).

This works because everything SO101PickEnv.reward_info() reads is in the 41-number obs.
The dream therefore needs no learned reward head: predict next_obs, then call this.
Consequence: every dream-vs-reality reward gap comes from DYNAMICS error, which keeps the analysis clean.

obs layout (see envs/so101_pick.py):
  0:6 qpos | 6:12 qvel | 12:15 claw | 15:18 box | 18:21 grasp_point - claw | 21:27 ctrl
  27:30 finger axis | 30:33 jaw axis | 33:36 box vel | 36:39 box up | 39 fixed-jaw touch | 40 moving-jaw touch
"""
import numpy as np

BOX_Z, LIFT_FULL, SQUEEZE_MIN, SUCCESS_LIFT = 0.03, 0.08, 0.05, 0.05


def pick_reward(next_obs, consistent=False):
    """next_obs: (..., 41) array -> (reward, grasped, success), each of shape (...).
    consistent=True: a grasp also requires the claw within 6 cm of the grasp point. In reality jaw contact
    implies that anyway (identical reward on real data); in the dream it blocks the dream_v5 exploit, where
    contact flags drifted to ~1.9 while the claw hovered 16 cm away."""
    o = np.asarray(next_obs)
    dist = np.linalg.norm(o[..., 18:21], axis=-1)
    reach = 1.0 - np.tanh(5.0 * dist)
    down = np.maximum(0.0, -o[..., 29])            # z-component of finger axis, sign flipped
    squeeze = o[..., 5] - o[..., 26]               # gripper joint angle - gripper motor target
    # contact flags are 0/1 in reality but a model predicts real numbers -> threshold at 0.5
    grasped = (o[..., 39] > 0.5) & (o[..., 40] > 0.5) & (squeeze > SQUEEZE_MIN)
    if consistent:
        grasped = grasped & (dist < 0.06)   # real grasps: max 5.1 cm (p99.9 4.8) over 91k grasps
    height = o[..., 17] - BOX_Z
    lift = np.clip(height / LIFT_FULL, 0.0, 1.0)
    reward = reach + 0.5 * down + grasped * (0.5 + 6.0 * lift)
    return reward, grasped, grasped & (height >= SUCCESS_LIFT)
