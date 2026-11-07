"""Train SAC entirely INSIDE the world model (the dream), then score it in real MuJoCo.

    python wm/dream_sac.py --name dream_v1 --wm outputs/wm_v1.pt --data data/pick_v1.npz

MBPO-style dream env:
  reset(): start from a REAL state sampled from the dataset (anchored to reality)
  step(a): a randomly chosen ensemble member predicts next_obs; reward = wm/reward.py(next_obs)
  after H=5 steps -> truncated (NOT terminated), so SAC's critic bootstraps the value beyond step 5.
  Short rollouts keep the dream inside its accurate regime (5-step error ~0.5 cm, see wm_v1.log).
  A random member per step: to exploit the dream, the policy must fool all 5 models at once.

Zero new real-environment steps are used for TRAINING. Real MuJoCo is only used to SCORE the policy.
Every 25k steps, on the same 20 unseen boxes:
  real : the policy runs 150 steps in MuJoCo                 (the truth)
  dream: the policy runs 150 steps in the world model alone   (what the dream believes)
  gap = dream - real: how much the dream is fooling the policy.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from wm.model import EnsembleDynamics  # noqa: E402
from wm.reward import pick_reward  # noqa: E402
from train.train_pick import evaluate as real_evaluate, report  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

torch.set_num_threads(1)


def load_wm(path):
    wm = EnsembleDynamics()
    wm.load_state_dict(torch.load(path))
    return wm.eval()


class DreamEnv(gym.Env):
    def __init__(self, wm, start_states, horizon=5, seed=0, penalty=0.0, cutoff=None, consistent=False):
        self.consistent = consistent
        # cutoff: end the dream (truncate) as soon as ensemble disagreement exceeds this -> the dream stops
        # the moment it starts guessing. In-distribution disagreement ~0.03, dream_v1's broken states ~0.45.
        self.cutoff, self.lengths = cutoff, []
        # penalty = MOPO's lambda: reward -= penalty * ensemble disagreement (std across the 5 members,
        # in units of the typical one-step change, averaged over the 41 dims). In-distribution it is ~0.03,
        # on dream_v1's out-of-distribution states ~0.45 -> the penalty hits exactly where the dream lies.
        self.wm, self.starts, self.H, self.penalty = wm, start_states, horizon, penalty
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(41,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        self.t = 0
        self.obs = self.starts[self.rng.integers(len(self.starts))].astype(np.float64)
        return self.obs.copy(), {}

    def step(self, action):
        a = torch.as_tensor(np.clip(action, -1, 1), dtype=torch.float32)[None]
        o = torch.as_tensor(self.obs, dtype=torch.float32)[None]
        member = self.rng.integers(self.wm.E)
        P = self.wm.predict(o, a)[:, 0]                     # (E,41): every member's prediction
        self.obs = P[member].numpy().astype(np.float64)
        r, grasped, success = pick_reward(self.obs, self.consistent)
        dis = float((P.std(0) / self.wm.d_sd).mean())
        if self.penalty:
            r = r - self.penalty * dis
        self.t += 1
        trunc = self.t >= self.H or (self.cutoff is not None and dis > self.cutoff)
        if trunc:
            self.lengths.append(self.t)
        return self.obs.copy(), float(r), False, trunc, {"is_success": bool(success)}


def dream_evaluate(model, vecnorm, wm, episodes=20, steps=150):
    """Full-length dream from the SAME start states as the real eval (seeds 10_000+i)."""
    env, succ, grasped, best, rets = SO101PickEnv(), 0, 0, [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        ever, top, ret = False, -1.0, 0.0
        for _ in range(steps):
            a = model.predict(vecnorm.normalize_obs(obs), deterministic=True)[0]
            obs = wm.predict(torch.as_tensor(obs, dtype=torch.float32)[None],
                             torch.as_tensor(a, dtype=torch.float32)[None]).mean(0)[0].numpy().astype(np.float64)
            r, g, s = pick_reward(obs)
            ever |= bool(g); top = max(top, obs[17] - 0.03); ret += float(r)
        succ += bool(s); grasped += ever; best.append(top); rets.append(ret)
    return succ / episodes, grasped / episodes, float(np.median(best)), float(np.mean(rets))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--wm", default="outputs/wm_v1.pt")
    ap.add_argument("--data", default="data/pick_v1.npz")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutoff", type=float, default=None, help="truncate the dream when disagreement > this")
    ap.add_argument("--penalty", type=float, default=0.0, help="MOPO uncertainty penalty lambda (0 = dream_v1 baseline)")
    args = ap.parse_args()
    print(vars(args), flush=True)

    wm = load_wm(ROOT / args.wm)
    starts = np.load(ROOT / args.data)["obs"]
    venv = vecnorm = VecNormalize(DummyVecEnv([lambda: DreamEnv(wm, starts, args.horizon, args.seed, args.penalty, args.cutoff)]),
                                  norm_obs=True, norm_reward=False)
    model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "outputs/tb"))
    t0, chunk = time.time(), 25_000
    for done in range(chunk, args.steps + 1, chunk):
        model.learn(chunk, reset_num_timesteps=False, tb_log_name=args.name)
        real = real_evaluate(model, vecnorm)
        dream = dream_evaluate(model, vecnorm, wm)
        report(f"{done // 1000}k REAL", real)
        report(f"{done // 1000}k DREAM", dream)
        L = venv.venv.envs[0].lengths; avg_len = np.mean(L) if L else float("nan"); L.clear()
        print(f"{'':>14}  avg dream length {avg_len:.1f} steps (max {args.horizon})")
        print(f"{'':>14}  gap (dream - real): success {dream[0] - real[0]:+.0%}  return {dream[3] - real[3]:+.1f}"
              f"   ({(time.time() - t0) / 60:.0f} min)", flush=True)
        model.save(ROOT / f"outputs/{args.name}")
        vecnorm.save(str(ROOT / f"outputs/{args.name}_vecnorm.pkl"))
