"""The decisive test: is the DREAM to blame, or the dream-training RECIPE?

    python wm/short_real.py --name short_real

Exactly dream_v1's recipe (SAC, 5-step episodes, each starting from a random dataset state, truncated
so the critic bootstraps), but the physics is REAL MuJoCo instead of the world model. The simulator is
restored to the exact state behind each dataset obs (qpos/qvel/ctrl saved by wm/collect.py).

  learns to pick here -> the recipe is fine, the world model's errors are what break dream_v1
  fails here too      -> the world model is innocent; short random-start episodes break learning
Scored exactly like everything else: 20 unseen boxes from the normal start pose, full 150 steps.
Ablation (which half of the recipe breaks learning?):
  --horizon 150 --starts data   full-length episodes, random dataset starts
  --horizon 5   --starts reset  5-step episodes, normal start pose
"""
import argparse, sys, time
from pathlib import Path
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from train.train_pick import evaluate, report  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

torch.set_num_threads(1)


class ShortRealEnv(SO101PickEnv):
    def __init__(self, states, horizon=5, seed=0, starts="data"):
        super().__init__()
        self.qp, self.qv, self.ct = states
        self.H, self.rng, self.starts = horizon, np.random.default_rng(seed), starts

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.t = 0
        if self.starts == "reset":                      # normal start pose, new random box
            return obs, info
        i = self.rng.integers(len(self.qp))
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = self.qp[i], self.qv[i], self.ct[i]
        mujoco.mj_forward(self.model, self.data)
        self.box_start, self.t = self.box_pos(), 0
        return self._get_obs(), {}

    def step(self, action):
        obs, r, term, _, info = super().step(action)
        self.t += 1
        return obs, r, term, self.t >= self.H, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="short_real")
    ap.add_argument("--data", default="data/pick_v2_state.npz")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", choices=["data", "reset"], default="data",
                    help="data = random dataset states (dream recipe); reset = normal start pose")
    args = ap.parse_args()
    print(vars(args), flush=True)
    D = np.load(ROOT / args.data)
    states = (D["qpos"], D["qvel"], D["ctrl"])

    # sanity: restoring a dataset state must reproduce its obs exactly
    env = ShortRealEnv(states, seed=1)
    env.rng = np.random.default_rng(123); i = np.random.default_rng(123).integers(len(D["qpos"]))
    o, _ = env.reset()
    print(f"restore check: max |obs - dataset obs| = {np.abs(o - D['obs'][i]).max():.1e}", flush=True)

    venv = vecnorm = VecNormalize(DummyVecEnv([lambda: ShortRealEnv(states, args.horizon, args.seed, args.starts)]),
                                  norm_obs=True, norm_reward=False)
    model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "outputs/tb"))
    t0, chunk = time.time(), 25_000
    for done in range(chunk, args.steps + 1, chunk):
        model.learn(chunk, reset_num_timesteps=False, tb_log_name=args.name)
        report(f"{done // 1000}k REAL", evaluate(model, vecnorm))
        print(f"{'':>14}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        model.save(ROOT / f"outputs/{args.name}")
        vecnorm.save(str(ROOT / f"outputs/{args.name}_vecnorm.pkl"))
