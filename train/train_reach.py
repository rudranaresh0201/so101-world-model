"""Train SAC on the SO-101 reach env.
    python train/train_reach.py                 # 60k steps, ~20 min on this laptop
    python train/train_reach.py --steps 20000   # quick run

Every 10k steps it evaluates on 30 fixed box positions it never trained on and prints:
  success rate, knocked rate, median final distance.  Saves to outputs/reach_sac.zip
Watch the learning curves live:  tensorboard --logdir outputs/tb
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

# Tiny networks + many CPU threads = threads fighting each other. 1 thread was ~7x faster here.
torch.set_num_threads(1)


def evaluate(model, episodes=30):
    env = SO101ReachEnv()
    success = knocked = 0
    final_dist = []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)   # fixed test boxes, never used in training
        done = False
        while not done:
            action = env.action_space.sample() if model is None else model.predict(obs, deterministic=True)[0]
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
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report("random policy", evaluate(None))
    model = SAC("MlpPolicy", SO101ReachEnv(), seed=args.seed, verbose=0,
                tensorboard_log=str(ROOT / "outputs/tb"))
    t0, chunk = time.time(), 10_000
    for done_steps in range(chunk, args.steps + 1, chunk):
        model.learn(chunk, reset_num_timesteps=False, tb_log_name=f"reach_sac_s{args.seed}")
        report(f"{done_steps // 1000}k steps", evaluate(model))
        print(f"{'':>14}  ({time.time() - t0:.0f}s elapsed)", flush=True)
        model.save(ROOT / "outputs/reach_sac")
    print("saved outputs/reach_sac.zip  ->  make a video: python train/watch.py")
