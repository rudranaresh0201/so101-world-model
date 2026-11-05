"""Train SAC on the SO-101 reach env.
    python train/train_reach.py --name myrun [--steps 60000] [--seed 0] [--norm-obs] [--no-knock-term]

Every 10k steps: evaluate on 30 fixed unseen boxes -> success rate, knocked rate, median final distance.
Saves outputs/<name>.zip (+ outputs/<name>_vecnorm.pkl if --norm-obs). Curves: tensorboard --logdir outputs/tb
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_reach import SO101ReachEnv  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

torch.set_num_threads(1)   # tiny nets: 1 thread was ~7x faster than the default here


def evaluate(model, vecnorm=None, episodes=30):
    env = SO101ReachEnv()   # evaluation always terminates on knocks, so runs are comparable
    success = knocked = 0
    final_dist = []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        done = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
                action = model.predict(o, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        success += info["is_success"]
        knocked += info["knocked"]
        final_dist.append(info["distance"])
    return success / episodes, knocked / episodes, float(np.median(final_dist))


def report(label, stats):
    s, k, d = stats
    print(f"{label:>14}  success {s:5.0%}   knocked {k:5.0%}   median final distance {100 * d:5.1f} cm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm-obs", action="store_true", help="normalise observations (running mean/std)")
    ap.add_argument("--no-knock-term", action="store_true", help="a knocked box does not end training episodes")
    args = ap.parse_args()
    print(vars(args), flush=True)

    venv = DummyVecEnv([lambda: Monitor(SO101ReachEnv(terminate_on_knock=not args.no_knock_term))])
    vecnorm = None
    if args.norm_obs:
        venv = vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=False)

    report("random policy", evaluate(None))
    model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "outputs/tb"))
    t0, chunk = time.time(), 10_000
    for done_steps in range(chunk, args.steps + 1, chunk):
        model.learn(chunk, reset_num_timesteps=False, tb_log_name=args.name)
        report(f"{done_steps // 1000}k steps", evaluate(model, vecnorm))
        print(f"{'':>14}  ({time.time() - t0:.0f}s elapsed)", flush=True)
        model.save(ROOT / f"outputs/{args.name}")
        if vecnorm is not None:
            vecnorm.save(str(ROOT / f"outputs/{args.name}_vecnorm.pkl"))
