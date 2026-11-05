"""Record the trained policy:  python train/watch.py  ->  outputs/reach.mp4 (4 episodes, new box each time)"""
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_reach import SO101ReachEnv  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

model = SAC.load(ROOT / "outputs/reach_sac")
env = SO101ReachEnv(render_mode="rgb_array")
frames = []
for ep in range(4):
    obs, _ = env.reset(seed=20_000 + ep)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
        frames.append(env.render())
        done = term or trunc
    frames += [frames[-1]] * 10   # pause on the final frame
    print(f"episode {ep}: {'SUCCESS' if info['is_success'] else 'miss'}  final distance {100 * info['distance']:.1f} cm")
iio.imwrite(ROOT / "outputs/reach.mp4", np.stack(frames), fps=20)
print("wrote outputs/reach.mp4")
