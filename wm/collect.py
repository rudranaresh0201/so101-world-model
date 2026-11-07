"""Collect real MuJoCo transitions for training the world model.

    python wm/collect.py --out data/pick_v1.npz

A world model only learns the physics it has SEEN. A dataset from the solved policy alone would teach
"the box always gets grasped" and nothing about misses, pushes or drops. So we mix behaviours:
  random actions         -> broad coverage of arm motion, box almost never touched
  pick v1 (stuck policy) -> lots of jaws resting on the box without squeezing
  pick v2 (solved) + action noise at 3 levels -> grasps, lifts, slips and drops
Each episode is tagged with its source, so we can later ask "where is the model wrong?" per source.

Saved arrays (N transitions, E episodes):
  obs (N,41)  act (N,6)  next_obs (N,41)  rew (N,)  ep (N,)  ep_source (E,) str
  qpos (N,13) qvel (N,12) ctrl (N,6): the exact MuJoCo state behind each obs (for wm/short_real.py)
Also checks that wm/reward.py reproduces the env reward exactly from next_obs.
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from wm.reward import pick_reward  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

torch.set_num_threads(1)

# (source label, checkpoint or None for random, action-noise std, episodes)
MIX = [
    ("random",       None,             0.0, 200),
    ("v1_noise0.3",  "pickv1_s1_225k", 0.3, 200),
    ("v2_noise0.0",  "pickv2_fresh",   0.0, 150),
    ("v2_noise0.3",  "pickv2_fresh",   0.3, 150),
    ("v2_noise0.6",  "pickv2_fresh",   0.6, 150),
    ("v2_noise1.0",  "pickv2_fresh",   1.0, 150),   # 0.6 still succeeded 100% in the smoke test: need real failures
    ("v2w_noise0.3", "pickv2_warm",    0.3, 150),
]


def load(name):
    model = SAC.load(ROOT / f"outputs/{name}.zip", device="cpu")   # explicit .zip: a same-named folder would shadow it
    vecnorm = pickle.load(open(ROOT / f"outputs/{name}_vecnorm.pkl", "rb"))
    return model, vecnorm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pick_v1.npz")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply episode counts (0.05 for a smoke test)")
    args = ap.parse_args()

    env, rng = SO101PickEnv(), np.random.default_rng(0)
    O, A, O2, R, EP, SRC, QP, QV, CT = [], [], [], [], [], [], [], [], []
    seed, t0, worst = 30_000, time.time(), 0.0       # seeds 30k+: disjoint from eval (10k) and video (20k)
    for label, ckpt, noise, n in MIX:
        policy = load(ckpt) if ckpt else None
        n = max(1, int(n * args.scale))
        succ = 0
        for _ in range(n):
            obs, _ = env.reset(seed=seed); seed += 1
            done = False
            while not done:
                if policy is None:
                    a = env.action_space.sample()
                else:
                    a = policy[0].predict(policy[1].normalize_obs(obs), deterministic=True)[0]
                    a = np.clip(a + noise * rng.standard_normal(6), -1, 1)
                QP.append(env.data.qpos.copy()); QV.append(env.data.qvel.copy()); CT.append(env.data.ctrl.copy())  # exact sim state of obs
                nxt, r, term, trunc, info = env.step(a)
                worst = max(worst, abs(float(pick_reward(nxt)[0]) - r))
                O.append(obs); A.append(a); O2.append(nxt); R.append(r); EP.append(len(SRC))
                obs, done = nxt, term or trunc
            succ += info["is_success"]
            SRC.append(label)
        print(f"{label:>13}: {n} episodes, final success {succ / n:4.0%}   ({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / args.out
    np.savez_compressed(out, obs=np.array(O, np.float32), act=np.array(A, np.float32),
                        next_obs=np.array(O2, np.float32), rew=np.array(R, np.float32),
                        ep=np.array(EP, np.int32), ep_source=np.array(SRC),
                        qpos=np.array(QP), qvel=np.array(QV), ctrl=np.array(CT))   # lets MuJoCo restart from any dataset state
    print(f"{len(R)} transitions, {len(SRC)} episodes -> {out}")
    print(f"reward-from-obs check: max |pick_reward(next_obs) - env reward| = {worst:.2e}  (should be ~0)")
