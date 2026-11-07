"""Train the ensemble world model.   python wm/train_wm.py --data data/pick_v1.npz --out outputs/wm_v1.pt

Split by EPISODE (10% held out), not by transition: neighbouring transitions in one episode are nearly
identical, so a random transition split would leak and make validation look falsely good.

Reports on held-out episodes:
  1-step error per obs group (normalised MSE: 1.0 = no better than predicting the average delta)
  contact-flag accuracy and grasp accuracy  <- where we expect the model to struggle
  OPEN-LOOP rollout error: start from a real state, feed the REAL actions, but let the model feed itself
  its own predictions for H steps. This is what a policy training in the dream experiences, and errors
  compound: a small 1-step error can become a big 20-step error.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wm.model import EnsembleDynamics  # noqa: E402
from wm.reward import pick_reward  # noqa: E402

GROUPS = {"joints q": slice(0, 6), "joint vel": slice(6, 12), "claw xyz": slice(12, 15), "box xyz": slice(15, 18),
          "ctrl": slice(21, 27), "box vel": slice(33, 36), "contacts": slice(39, 41)}


def evaluate(model, val, ep_len=150, horizons=(1, 5, 20, 50)):
    o, a, o2 = (torch.from_numpy(val[k]) for k in ("obs", "act", "next_obs"))
    pred = model.predict(o, a).mean(0)                       # ensemble mean, (B,41)
    d_true, d_pred = (o2 - o - model.d_mu) / model.d_sd, (pred - o - model.d_mu) / model.d_sd
    print("  1-step normalised MSE:", "  ".join(f"{k} {((d_pred - d_true)[:, s] ** 2).mean():.3f}" for k, s in GROUPS.items()))
    p, t = pred.numpy(), val["next_obs"]
    flag_acc = ((p[:, 39:41] > 0.5) == (t[:, 39:41] > 0.5)).mean()
    g_pred, g_true = pick_reward(p)[1], pick_reward(t)[1]
    print(f"  contact-flag acc {flag_acc:.3f}   grasp: acc {(g_pred == g_true).mean():.3f}  "
          f"recall {(g_pred & g_true).sum() / max(1, g_true.sum()):.3f}  false-grasp rate "
          f"{(g_pred & ~g_true).sum() / max(1, (~g_true).sum()):.3f}")
    # open-loop rollouts from the start of each held-out episode
    starts = np.arange(0, len(val["obs"]), ep_len)
    s = torch.from_numpy(val["obs"][starts])
    errs, rew_gap = {}, {}
    for h in range(max(horizons)):
        idx = starts + h
        s = model.predict(s, torch.from_numpy(val["act"][idx])).mean(0)
        if h + 1 in horizons:
            real = val["next_obs"][idx]
            errs[h + 1] = (np.linalg.norm(s[:, 12:15].numpy() - real[:, 12:15], axis=1).mean() * 100,
                           np.linalg.norm(s[:, 15:18].numpy() - real[:, 15:18], axis=1).mean() * 100)
            rew_gap[h + 1] = np.abs(pick_reward(s.numpy())[0] - val["rew"][idx]).mean()
    print("  open-loop (real actions, model feeds itself):  " + "  ".join(
        f"H={h}: claw {c:.1f}cm box {b:.1f}cm |r err| {rew_gap[h]:.2f}" for h, (c, b) in errs.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pick_v1.npz")
    ap.add_argument("--out", default="outputs/wm_v1.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    D = dict(np.load(ROOT / args.data))
    n_ep = len(D["ep_source"])
    val_eps = rng.permutation(n_ep)[: n_ep // 10]
    is_val = np.isin(D["ep"], val_eps)
    keys = ("obs", "act", "next_obs", "rew")
    tr = {k: D[k][~is_val] for k in keys}
    val = {k: D[k][is_val] for k in keys}                    # stays ordered by episode -> rollouts work
    print(f"train {len(tr['obs'])} transitions, val {len(val['obs'])} ({len(val_eps)} episodes)")

    model = EnsembleDynamics()
    T = {k: torch.from_numpy(v) for k, v in tr.items()}
    model.set_stats(T["obs"], T["act"], T["next_obs"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    N, B, E = len(T["obs"]), 256, model.E
    target = (T["next_obs"] - T["obs"] - model.d_mu) / model.d_sd
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        boot = torch.from_numpy(rng.integers(0, N, size=(E, N)))   # bootstrap: each member its own resample
        total = 0.0
        for i in range(0, N, B):
            idx = boot[:, i:i + B]                                    # (E,B)
            pred = model.forward_norm(T["obs"][idx], T["act"][idx])
            loss = ((pred - target[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * idx.shape[1]
        sched.step()
        print(f"epoch {epoch:3d}  train loss {total / N:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if epoch % 10 == 0 or epoch == args.epochs:
            evaluate(model, val)
            torch.save(model.state_dict(), ROOT / args.out)
    print(f"saved {ROOT / args.out}")
