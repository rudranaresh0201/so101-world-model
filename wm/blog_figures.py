"""Figures for the blog post, from the training logs and one traced dream.

    MUJOCO_GL=egl python wm/blog_figures.py OUTDIR

fig_baseline.png  SAC in MuJoCo: pick v1 reward (touching pays) vs pick v2 (squeeze + lift)
fig_horizon.png   the rollout recipe run in REAL MuJoCo at horizons 5 / 20 / 50 / 150
fig_cheat.png     one dream_v5 episode traced: what the model believes vs what really happens
fig_exploit.png   dream_v5 over training: imagined lift vs real lift
fig_fixes.png     dream_v7 (3 seeds) and the ablation without the 6 cm rule, real success
"""
import pickle, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from wm.dream_sac import load_wm  # noqa: E402
from wm.reward import pick_reward  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs/blog")
OUT.mkdir(parents=True, exist_ok=True)

INK, DIM, FAINT, ACCENT, GREY, BG = "#1c1a17", "#6b6559", "#b9b1a2", "#c2410c", "#8a8375", "#fffdf8"
plt.rcParams.update({
    "font.family": "Ubuntu Sans", "font.size": 11, "axes.edgecolor": FAINT, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.titlesize": 12,
    "axes.titleweight": "medium", "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def curve(name, kind=None, col=2):
    """(steps_k, value) from a log. kind: 'REAL' / 'DREAM' / None (plain SAC logs). col 2 success, 3 grasped, 4 lift."""
    xs, ys = [], []
    for line in open(ROOT / f"outputs/{name}.log"):
        m = re.match(r"\s*(\d+)k (REAL|DREAM|steps)\s+success\s+(\d+)%\s+ever-grasped\s+(\d+)%\s+median best lift\s+(-?[\d.]+) cm", line)
        if m and (kind is None or m[2] == kind):
            xs.append(int(m[1])); ys.append(float(m[col + 1]))
    return np.array(xs), np.array(ys)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig); print("wrote", OUT / name)


# 1. baseline
fig, ax = plt.subplots(figsize=(7, 3.4))
x, y = curve("pick_s1"); ax.plot(x, y, color=GREY, lw=2, label="reward v1: touching the box pays")
x, y = curve("pickv2_fresh"); ax.plot(x, y, color=ACCENT, lw=2.4, label="reward v2: must squeeze, lifting pays most")
ax.set(xlabel="training steps (thousands)", ylabel="pick success (%)", ylim=(-3, 103))
ax.set_title("SAC in MuJoCo, 20 unseen box positions per point"); ax.legend(loc="center right")
save(fig, "fig_baseline.png")

# 2. horizon sweep in real physics
fig, ax = plt.subplots(figsize=(7, 3.4))
for name, lab, c, w in [("short_real", "5 steps", FAINT, 2), ("abl_h20", "20 steps", GREY, 2),
                        ("abl_h50", "50 steps", ACCENT, 2.4), ("abl_long_data", "150 steps (full episode)", INK, 1.6)]:
    x, y = curve(name, "REAL"); ax.plot(x, y, color=c, lw=w, label=lab)
ax.set(xlabel="training steps (thousands)", ylabel="pick success (%)", ylim=(-3, 103))
ax.set_title("Same short-episode recipe, but in real MuJoCo: only the episode length changes")
ax.legend(loc="upper left")
save(fig, "fig_horizon.png")

# 3. one traced dream: dream_v5 policy, first test box
torch.set_num_threads(2)
wm = load_wm(ROOT / "outputs/wm_v1.pt")
m = SAC.load(ROOT / "outputs/dream_v5.zip", device="cpu")
vn = pickle.load(open(ROOT / "outputs/dream_v5_vecnorm.pkl", "rb"))
real = SO101PickEnv(); o, _ = real.reset(seed=10_000); d = o.copy()
T = {k: [] for k in ("dist_d", "dist_r", "flag_d", "lift_d", "lift_r", "dis", "grasp_d")}
for t in range(150):
    o, *_ = real.step(m.predict(vn.normalize_obs(o), deterministic=True)[0])
    a = m.predict(vn.normalize_obs(d), deterministic=True)[0]
    P = wm.predict(torch.tensor(d, dtype=torch.float32)[None], torch.tensor(a, dtype=torch.float32)[None])[:, 0].numpy()
    d = P.mean(0).astype(np.float64)
    T["dist_d"].append(100 * np.linalg.norm(d[18:21])); T["dist_r"].append(100 * np.linalg.norm(o[18:21]))
    T["flag_d"].append(min(d[39], d[40])); T["lift_d"].append(100 * (d[17] - 0.03)); T["lift_r"].append(100 * (o[17] - 0.03))
    T["dis"].append(float(np.linalg.norm(P.std(0)))); T["grasp_d"].append(bool(pick_reward(d)[1]))
T = {k: np.array(v) for k, v in T.items()}
g0 = int(np.argmax(T["grasp_d"])) if T["grasp_d"].any() else None
print(f"trace: first imagined grasp at step {g0}, claw-box {T['dist_d'][g0]:.1f} cm (dream), final lift dream {T['lift_d'][-1]:.1f} real {T['lift_r'][-1]:.1f}")

fig, axs = plt.subplots(3, 1, figsize=(7, 6.4), sharex=True)
t = np.arange(1, 151)
axs[0].plot(t, T["lift_d"], color=ACCENT, lw=2.2, label="world model"); axs[0].plot(t, T["lift_r"], color=INK, lw=1.6, label="MuJoCo")
axs[0].set(ylabel="box height (cm)"); axs[0].set_title("One test episode: what the world model believes vs what happens")
axs[0].legend(loc="upper left")
axs[1].plot(t, T["flag_d"], color=ACCENT, lw=2.2); axs[1].axhline(0.5, color=FAINT, ls="--", lw=1)
axs[1].text(2, 0.56, "counted as touching above 0.5", color=DIM, fontsize=9)
axs[1].set(ylabel="contact signal\n(both jaws, model)")
axs[2].plot(t, T["dist_d"], color=ACCENT, lw=2.2); axs[2].plot(t, T["dist_r"], color=INK, lw=1.6)
axs[2].axhline(6, color=FAINT, ls="--", lw=1); axs[2].text(2, 7, "6 cm: no real grasp is ever farther", color=DIM, fontsize=9)
axs[2].set(ylabel="claw to box (cm)", xlabel="step (0.05 s each)")
if g0 is not None:
    for ax in axs:
        ax.axvline(g0 + 1, color=ACCENT, alpha=0.25, lw=6)
save(fig, "fig_cheat.png")

# 4. the exploit over training: imagined vs real lift
fig, ax = plt.subplots(figsize=(7, 3.4))
x, y = curve("dream_v5", "DREAM", 4); ax.plot(x, y, color=ACCENT, lw=2.4, marker="o", ms=4, label="lift the world model reports")
x, y = curve("dream_v5", "REAL", 4); ax.plot(x, y, color=INK, lw=1.8, marker="o", ms=4, label="lift in MuJoCo")
ax.set(xlabel="dream training steps (thousands)", ylabel="median best lift (cm)")
ax.set_title("Trained only in the model with 50-step rollouts"); ax.legend(loc="center right")
save(fig, "fig_exploit.png")

# 5. the fixes, real success
fig, ax = plt.subplots(figsize=(7, 3.4))
for name, lab, c, w in [("dream_v7_s1", "all fixes, seed 1", ACCENT, 2.4), ("dream_v7", "all fixes, seed 0", "#e08a5c", 2),
                        ("dream_v7_s2", "all fixes, seed 2", GREY, 2), ("dream_v7n", "fixes without the 6 cm rule", FAINT, 2)]:
    x, y = curve(name, "REAL"); ax.plot(x, y, color=c, lw=w, label=lab)
ax.set(xlabel="dream training steps (thousands)", ylabel="real pick success (%)", ylim=(-3, 103))
ax.set_title("Trained mostly in the model, tested in MuJoCo"); ax.legend(loc="upper left", ncol=2)
save(fig, "fig_fixes.png")
