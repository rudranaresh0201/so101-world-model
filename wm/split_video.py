"""Split-screen video: what the robot DREAMS (left) vs what REALLY happens (right).

    MUJOCO_GL=egl python wm/split_video.py      -> outputs/dream_vs_reality.mp4

Both sides start from the same box and run the same policy for 150 steps.
  right: real MuJoCo physics
  left : the world model alone. Each predicted state (joint angles + box position/tilt) is drawn by
         putting MuJoCo's robot into that pose, so the left side shows exactly what the dream believes.
Part 1: dream_v5 (no fixes)          -> the dream hallucinates a grasp and the box floats away
Part 2: dream_v7 seed 1 @125k (fixes) -> the dream-trained policy picks the box for real
"""
import pickle, sys
from pathlib import Path
import imageio.v3 as iio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from wm.model import EnsembleDynamics  # noqa: E402
from wm.reward import pick_reward  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

FONT = ImageFont.load_default(size=18)
BIG = ImageFont.load_default(size=26)


def load(policy, wm_path):
    m = SAC.load(ROOT / policy, device="cpu")
    vn = pickle.load(open(ROOT / policy.replace(".zip", "_vecnorm.pkl"), "rb"))
    wm = EnsembleDynamics(); wm.load_state_dict(torch.load(ROOT / wm_path)); wm.eval()
    return m, vn, wm


def quat_from_up(u):
    """Box orientation from its up-axis (yaw unknown -> 0). Good enough to SHOW a tilt."""
    u = u / (np.linalg.norm(u) + 1e-9); z = np.array([0, 0, 1.0])
    axis = np.cross(z, u); s = np.linalg.norm(axis); c = float(np.clip(z @ u, -1, 1))
    if s < 1e-6:
        return np.array([1.0, 0, 0, 0])
    half = np.arccos(c) / 2
    return np.concatenate([[np.cos(half)], axis / s * np.sin(half)])


def show_dream_state(env, o):
    d = env.data
    d.qpos[:6] = o[0:6]
    a = env.box_qpos_adr
    d.qpos[a:a + 3] = o[15:18]
    d.qpos[a + 3:a + 7] = quat_from_up(o[36:39])
    mujoco.mj_forward(env.model, d)
    return env.render()


def label(frame, text, sub, color):
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 54], fill=(0, 0, 0))
    dr.text((10, 4), text, font=BIG, fill=color)
    dr.text((10, 32), sub, font=FONT, fill=(230, 230, 230))
    return np.asarray(im)


def episode(m, vn, wm, seed, consistent):
    real, dream = SO101PickEnv(render_mode="rgb_array"), SO101PickEnv(render_mode="rgb_array")
    o, _ = real.reset(seed=seed); dream.reset(seed=seed); d = o.copy()
    frames, r_ret, d_ret = [], 0.0, 0.0
    for t in range(150):
        a_r = m.predict(vn.normalize_obs(o), deterministic=True)[0]
        a_d = m.predict(vn.normalize_obs(d), deterministic=True)[0]
        o, r, *_ , info = real.step(a_r); r_ret += r
        d = wm.predict(torch.tensor(d, dtype=torch.float32)[None], torch.tensor(a_d, dtype=torch.float32)[None]).mean(0)[0].numpy().astype(np.float64)
        dr, dg, _ = pick_reward(d, consistent); d_ret += float(dr)
        L = label(show_dream_state(dream, d), "DREAM (world model)",
                  f"box height {100 * (d[17] - 0.03):5.1f} cm   grasped={bool(dg)}", (120, 200, 255))
        R = label(real.render(), "REALITY (MuJoCo physics)",
                  f"box height {100 * info['height']:5.1f} cm   grasped={info['grasped']}", (255, 210, 90))
        frames.append(np.concatenate([L, np.full((L.shape[0], 6, 3), 255, np.uint8), R], axis=1))
    return frames, d_ret, r_ret, info["is_success"], float(d[17] - 0.03)


def title_card(text, sub, shape, n=40):
    im = Image.new("RGB", (shape[1], shape[0]), (15, 15, 20)); dr = ImageDraw.Draw(im)
    dr.text((30, shape[0] // 2 - 40), text, font=BIG, fill=(255, 255, 255))
    for i, line in enumerate(sub.split("\n")):
        dr.text((30, shape[0] // 2 + 4 + 24 * i), line, font=FONT, fill=(200, 200, 200))
    return [np.asarray(im)] * n


if __name__ == "__main__":
    parts = [
        ("Part 1: trained only in the dream (no fixes)",
         "The dream believes the robot grasped the box and lifted it.\nReality: the claw hovers ~16 cm away. The box never moves.",
         "outputs/dream_v5.zip", "outputs/wm_v1.pt", False, "exploit"),
        ("Part 2: dream + fixes (real-data loop, penalty, 6 cm rule)",
         "The policy still learns mostly in the dream, and now it picks\nthe box in reality (about half the boxes at this checkpoint).",
         "outputs/ckpt/dream_v7_s1_125k.zip", "outputs/ckpt/dream_v7_s1_125k_wm.pt", True, "success"),
    ]
    video = []
    for title, sub, pol, wmp, cons, want in parts:
        m, vn, wm = load(pol, wmp)
        chosen = []
        for seed in range(10_000, 10_040):
            fr, dret, rret, ok, dh = episode(m, vn, wm, seed, cons)
            good = (dh > 0.10 and not ok) if want == "exploit" else ok
            print(f"{title[:6]} seed {seed}: dream lift {100*dh:5.1f}cm real success {ok}  -> {'KEEP' if good else 'skip'}", flush=True)
            if good:
                chosen.append(fr)
            if len(chosen) == 2:
                break
        video += title_card(title, sub, chosen[0][0].shape)
        for fr in chosen:
            video += fr + [fr[-1]] * 15
    out = ROOT / "outputs/dream_vs_reality.mp4"
    iio.imwrite(out, np.stack(video), fps=20, codec="libx264", macro_block_size=1, pixelformat="yuv420p")
    print("wrote", out)
