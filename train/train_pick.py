"""Train SAC on the SO-101 pick env.   python train/train_pick.py --name P1 [--steps 600000] [--seed 0]
Every 25k steps: 20 unseen boxes -> success rate, ever-grasped rate, median best lift, mean return.
Saves outputs/<name>.zip + outputs/<name>_vecnorm.pkl. Curves: tensorboard --logdir outputs/tb"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

torch.set_num_threads(1)


def evaluate(model, vecnorm, episodes=20):
    env = SO101PickEnv()
    succ, grasped, best_lift, rets = 0, 0, [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        done, ever, top, ret = False, False, -1.0, 0.0
        while not done:
            a = env.action_space.sample() if model is None else model.predict(vecnorm.normalize_obs(obs), deterministic=True)[0]
            obs, r, term, trunc, info = env.step(a)
            ever |= info["grasped"]; top = max(top, info["height"]); ret += r
            done = term or trunc
        succ += info["is_success"]; grasped += ever; best_lift.append(top); rets.append(ret)
    return succ / episodes, grasped / episodes, float(np.median(best_lift)), float(np.mean(rets))


def report(label, s):
    print(f"{label:>14}  success {s[0]:4.0%}  ever-grasped {s[1]:4.0%}  median best lift {100*s[2]:5.1f} cm  return {s[3]:6.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="warm-start from outputs/<init>.zip + its vecnorm stats")
    args = ap.parse_args()
    print(vars(args), flush=True)
    venv = vecnorm = VecNormalize(DummyVecEnv([lambda: Monitor(SO101PickEnv())]), norm_obs=True, norm_reward=False)
    report("random policy", evaluate(None, vecnorm))
    if args.init:
        venv = vecnorm = VecNormalize.load(str(ROOT / f"outputs/{args.init}_vecnorm.pkl"), venv.venv)
        model = SAC.load(ROOT / f"outputs/{args.init}", env=venv, seed=args.seed, tensorboard_log=str(ROOT / "outputs/tb"))
    else:
        model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "outputs/tb"))
    t0, chunk = time.time(), 25_000
    for done_steps in range(chunk, args.steps + 1, chunk):
        model.learn(chunk, reset_num_timesteps=False, tb_log_name=args.name)
        report(f"{done_steps // 1000}k steps", evaluate(model, vecnorm))
        print(f"{'':>14}  ({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        model.save(ROOT / f"outputs/{args.name}")
        vecnorm.save(str(ROOT / f"outputs/{args.name}_vecnorm.pkl"))
