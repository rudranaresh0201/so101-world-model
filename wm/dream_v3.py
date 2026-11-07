"""dream_v3: MBPO-style ONLINE loop. Dream training plus a small, steady trickle of real data.

    python wm/dream_v3.py --name dream_v3

Why (from dream_v1/v2, see memory/notes): fully offline, the policy never learned to reach even INSIDE the
dream. (1) Half the start states were "already holding the box", so it rarely practised approaching.
(2) It never trained on the states IT gets itself into, so in reality it drifted somewhere unfamiliar.

The loop, every --every dream steps:
  1. run the CURRENT policy (stochastic, so it explores) for --real_eps episodes in real MuJoCo
  2. add those transitions to the real dataset, fine-tune the world model 1 pass  -> the dream gets
     corrected exactly where the policy now goes
  3. dream start states: 50% from the policy's OWN real episodes, 50% from the original dataset
Everything else is as in dream_v1 (5-step rollouts, random ensemble member, no penalty), so the
comparison v1 -> v3 isolates the real-data loop. Real steps used are counted and printed: that's the
budget for the headline comparison against SAC trained purely in reality (225k+ real steps to take off).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from wm.dream_sac import DreamEnv, load_wm, dream_evaluate  # noqa: E402
from train.train_pick import evaluate as real_evaluate, report  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402


class MixedStartDream(DreamEnv):
    """Start states: half from the policy's own real episodes, half from the original dataset."""
    def __init__(self, wm, base_starts, seed=0, horizon=5, penalty=0.0, consistent=False):
        super().__init__(wm, base_starts, horizon=horizon, seed=seed, penalty=penalty, consistent=consistent)
        self.online = None

    def reset(self, seed=None, options=None):
        self.t = 0
        pool = self.online if (self.online is not None and self.rng.random() < 0.5) else self.starts
        self.obs = pool[self.rng.integers(len(pool))].astype(np.float64)
        return self.obs.copy(), {}


def collect_real(model, vecnorm, env, episodes, seed):
    O, A, O2, R, T = [], [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            a = model.predict(vecnorm.normalize_obs(obs), deterministic=False)[0]   # stochastic: explore
            nxt, r, term, trunc, _ = env.step(a)
            O.append(obs); A.append(a); O2.append(nxt); R.append(r); T.append(trunc)
            obs, done = nxt, term or trunc
    return np.array(O, np.float32), np.array(A, np.float32), np.array(O2, np.float32), np.array(R), np.array(T)


def add_real_to_sac(model, o, a, o2, r, trunc):
    """MBPO's 'real ratio': real transitions also go into the POLICY's replay buffer, so the critic sees
    true rewards, not only dream rewards. SB3 stores raw obs and normalises at sampling time."""
    for i in range(len(o)):
        model.replay_buffer.add(o[i:i + 1], o2[i:i + 1], a[i:i + 1], np.array([r[i]]), np.array([bool(trunc[i])]),
                                [{"TimeLimit.truncated": bool(trunc[i])}])


def finetune(wm, O, A, O2, opt, rng, recent, batch=256):
    """One pass over the data. Half of each batch comes from the most recent real episodes, so new
    data (where the policy is now) actually moves the model instead of drowning in the old 172k."""
    N, E = len(O), wm.E
    target = (O2 - O - wm.d_mu) / wm.d_sd
    for _ in range(N // (4 * batch)):                       # a quarter-epoch: cheap, frequent updates
        old = rng.integers(0, N, size=(E, batch // 2))
        new = rng.integers(max(0, N - recent), N, size=(E, batch - batch // 2))
        idx = torch.from_numpy(np.concatenate([old, new], 1))
        loss = ((wm.forward_norm(O[idx], A[idx]) - target[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dream_v3")
    ap.add_argument("--wm", default="outputs/wm_v1.pt")
    ap.add_argument("--data", default="data/pick_v1.npz")
    ap.add_argument("--steps", type=int, default=300_000, help="dream steps")
    ap.add_argument("--every", type=int, default=5_000, help="dream steps between real-data rounds")
    ap.add_argument("--real_eps", type=int, default=4, help="real episodes per round (150 steps each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--penalty", type=float, default=0.0, help="MOPO uncertainty penalty lambda")
    ap.add_argument("--consistent", action="store_true", help="grasp requires claw within 6 cm (blocks the v5 exploit)")
    ap.add_argument("--real_mix", action="store_true", help="also put real transitions into SAC's replay buffer")
    ap.add_argument("--horizon", type=int, default=5, help="dream length (the sweep: 5 fails, 20 slow, 50 works)")
    args = ap.parse_args()
    print(vars(args), flush=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)

    wm = load_wm(ROOT / args.wm)
    for p in wm.parameters():
        p.requires_grad_(True)
    D = np.load(ROOT / args.data)
    O, A, O2 = (torch.from_numpy(D[k]) for k in ("obs", "act", "next_obs"))
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)

    dream = MixedStartDream(wm, D["obs"], args.seed, args.horizon, args.penalty, args.consistent)
    venv = vecnorm = VecNormalize(DummyVecEnv([lambda: dream]), norm_obs=True, norm_reward=False)
    model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "outputs/tb"))
    real_env, real_steps, online, t0 = SO101PickEnv(), 0, [], time.time()
    real_seed = 50_000                                    # disjoint from eval (10k), video (20k), dataset (30k)

    for done in range(args.every, args.steps + 1, args.every):
        model.learn(args.every, reset_num_timesteps=False, tb_log_name=args.name)
        # 1. real episodes with the current policy
        o, a, o2, r_real, tr = collect_real(model, vecnorm, real_env, args.real_eps, real_seed)
        if args.real_mix:
            add_real_to_sac(model, o, a, o2, r_real, tr)
        real_seed += args.real_eps; real_steps += len(o)
        online.append(o)
        # 2. fold them into the world model's data and fine-tune
        O, A, O2 = torch.cat([O, torch.from_numpy(o)]), torch.cat([A, torch.from_numpy(a)]), torch.cat([O2, torch.from_numpy(o2)])
        wm.train(); loss = finetune(wm, O, A, O2, opt, rng, recent=20 * 150); wm.eval()
        # 3. dream starts now include the policy's own states
        dream.online = np.concatenate(online)
        if done % 25_000 == 0:
            real = real_evaluate(model, vecnorm)
            dr = dream_evaluate(model, vecnorm, wm)
            report(f"{done // 1000}k REAL", real)
            report(f"{done // 1000}k DREAM", dr)
            print(f"{'':>14}  real steps used so far: {real_steps}   wm loss {loss:.3f}   "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            model.save(ROOT / f"outputs/{args.name}")
            vecnorm.save(str(ROOT / f"outputs/{args.name}_vecnorm.pkl"))
            torch.save(wm.state_dict(), ROOT / f"outputs/{args.name}_wm.pt")
            tag = ROOT / f"outputs/ckpt/{args.name}_{done // 1000}k"      # keep EVERY checkpoint (v7 lost its 30% policy)
            model.save(f"{tag}.zip"); vecnorm.save(f"{tag}_vecnorm.pkl"); torch.save(wm.state_dict(), f"{tag}_wm.pt")
