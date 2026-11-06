"""Tweet-quality video of a trained policy: 1080p, 40 fps, close camera that follows the box.
    python train/render_hd.py --name D_long_s1 --env reach
    python train/render_hd.py --name pick_s1 --env pick --episodes 3
"""
import argparse, pickle, sys
from pathlib import Path
import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_reach import SO101ReachEnv  # noqa: E402
from envs.so101_pick import SO101PickEnv  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--env", choices=["reach", "pick"], default="pick")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--azimuth", type=float, default=150)
args = ap.parse_args()

model = SAC.load(ROOT / f"outputs/{args.name}")
vecnorm = pickle.load(open(ROOT / f"outputs/{args.name}_vecnorm.pkl", "rb")); vecnorm.training = False
env = (SO101PickEnv if args.env == "pick" else SO101ReachEnv)()
m, d = env.model, env.data
m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height   # allow a big framebuffer
m.vis.quality.shadowsize = 8192                                              # crisp shadows
renderer = mujoco.Renderer(m, height=args.height, width=args.width)
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 0.55, args.azimuth, -22

frames = []
real_step = mujoco.mj_step
def step_and_film(mm, dd, n=[0]):           # film every 5 physics steps -> 40 fps, smoother than 20
    real_step(mm, dd); n[0] += 1
    if n[0] % 5 == 0:
        target = 0.5 * (env.box_pos() + np.array([0.12, 0.0, 0.08]))       # between robot and box
        cam.lookat[:] = 0.9 * cam.lookat + 0.1 * target                     # smooth follow
        renderer.update_scene(dd, cam); frames.append(renderer.render())
mujoco.mj_step = step_and_film

for ep in range(args.episodes):
    obs, _ = env.reset(seed=20_000 + ep)
    cam.lookat[:] = 0.5 * (env.box_pos() + np.array([0.12, 0.0, 0.08]))
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(model.predict(vecnorm.normalize_obs(obs), deterministic=True)[0])
        done = term or trunc
    frames += [frames[-1]] * 20
    print(f"episode {ep}: {'SUCCESS' if info['is_success'] else 'miss'}", flush=True)

out = ROOT / f"outputs/{args.name}_hd.mp4"
iio.imwrite(out, np.stack(frames), fps=40, codec="libx264", quality=9, macro_block_size=8)
print("wrote", out)
