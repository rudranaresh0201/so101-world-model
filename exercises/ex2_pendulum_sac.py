"""Stage 2 exercise: SAC on a MuJoCo task with CONTINUOUS actions. ~10+ min on your CPU (SAC is slower per step than PPO).
    python exercises/ex2_pendulum_sac.py              # InvertedPendulum (easy)
    python exercises/ex2_pendulum_sac.py Reacher-v5   # a 2-joint arm reaching a target: closest to our project

Saves a video of the trained policy to outputs/.

TODO for you:
  1. Before training, print env.observation_space and env.action_space. What does each number mean?
     (Gymnasium docs page for the env lists them.)
  2. Reacher's reward = -distance_to_target - 0.1*|action|^2 ... find it in gymnasium/envs/mujoco/reacher_v5.py.
     Why penalise the action size?
  3. Compare PPO vs SAC on Reacher with the same step budget. Which learns faster? Why (off-policy replay)?
"""
import sys

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

env_id = sys.argv[1] if len(sys.argv) > 1 else "InvertedPendulum-v5"
steps = 10_000 if "Pendulum" in env_id else 50_000
env = gym.make(env_id)
print("observation space:", env.observation_space, "\naction space:", env.action_space)

model = SAC("MlpPolicy", env, verbose=0, seed=0, tensorboard_log="outputs/tb")
model.learn(total_timesteps=steps, tb_log_name=f"sac_{env_id}", progress_bar=True)
mean, std = evaluate_policy(model, env, n_eval_episodes=10)
print(f"{env_id} SAC: mean return {mean:.1f} +/- {std:.1f}")

venv = gym.make(env_id, render_mode="rgb_array")
frames, (obs, _) = [], venv.reset(seed=1)
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = venv.step(action)
    frames.append(venv.render())
    if term or trunc:
        obs, _ = venv.reset()
iio.imwrite(f"outputs/{env_id}.mp4", np.stack(frames), fps=30)
print(f"video: outputs/{env_id}.mp4")
