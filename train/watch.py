"""Record a trained policy:  python train/watch.py --name D_long_s1  ->  outputs/<name>.mp4 (6 episodes, unseen boxes)"""
import argparse
import pickle
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_reach import SO101ReachEnv  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--episodes", type=int, default=6)
args = ap.parse_args()

model = SAC.load(ROOT / f"outputs/{args.name}")
stats = ROOT / f"outputs/{args.name}_vecnorm.pkl"
vecnorm = pickle.load(open(stats, "rb")) if stats.exists() else None   # same obs scaling the policy trained with
if vecnorm is not None:
    vecnorm.training = False

env = SO101ReachEnv(render_mode="rgb_array")
frames, wins = [], 0
for ep in range(args.episodes):
    obs, _ = env.reset(seed=20_000 + ep)
    done = False
    while not done:
        o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        obs, _, term, trunc, info = env.step(model.predict(o, deterministic=True)[0])
        frames.append(env.render())
        done = term or trunc
    frames += [frames[-1]] * 10   # pause on the final frame
    wins += info["is_success"]
    print(f"episode {ep}: {'SUCCESS' if info['is_success'] else 'miss'}  final distance {100 * info['distance']:.1f} cm"
          f"{'  (knocked)' if info['knocked'] else ''}")
out = ROOT / f"outputs/{args.name}.mp4"
iio.imwrite(out, np.stack(frames), fps=20)
print(f"{wins}/{args.episodes} success -> wrote {out}")
