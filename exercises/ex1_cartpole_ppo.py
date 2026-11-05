"""Stage 1 exercise: PPO on CartPole (discrete actions). ~1-2 min on CPU.
    python exercises/ex1_cartpole_ppo.py
Then:  tensorboard --logdir outputs/tb   and look at rollout/ep_rew_mean.

TODO for you (do these, then explain the results out loud):
  1. Change total_timesteps to 5_000. What happens to the final score? Why?
  2. Set ent_coef=0.1. What happens? (Hint: entropy = exploration.)
  3. Read the SB3 PPO source: stable_baselines3/ppo/ppo.py, method train().
     Find the clipped objective. Which line is the "clip"?
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0, seed=0, tensorboard_log="outputs/tb")
model.learn(total_timesteps=50_000, tb_log_name="cartpole_ppo")
mean, std = evaluate_policy(model, env, n_eval_episodes=20)
print(f"CartPole PPO: mean episode return {mean:.0f} +/- {std:.0f}   (max possible: 500)")
