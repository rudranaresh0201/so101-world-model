"""Shareable story video (1280x720), for people who have never heard of RL.

    MUJOCO_GL=egl python wm/story_video.py            -> outputs/robot_story.mp4
    MUJOCO_GL=egl python wm/story_video.py --stills   -> framing check only (scratch PNGs)

Story: question -> it works (normal training) -> the idea (practise in imagination) -> the cheat
       -> the fix -> where it stands.
Clips use the first test boxes showing the behaviour (seeds 10000+); the printout lists them.
The imagination side is drawn by putting MuJoCo's robot into each state the world model predicts,
then tinted blue so the two worlds read differently without labels.
"""
import argparse, pickle, sys
from pathlib import Path
import imageio.v3 as iio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so101_pick import SO101PickEnv  # noqa: E402
from wm.split_video import load, quat_from_up  # noqa: E402
from wm.reward import pick_reward  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

W, H, FPS = 1280, 720, 20
VW, VH, GAP, TOP = 632, 560, 16, 24           # two views side by side; caption area below
BG = (11, 12, 16)
FONT = "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf"


def font(size, weight="Regular"):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F_LABEL, F_CAP, F_SUB, F_TITLE, F_BODY = font(22, "Bold"), font(34, "Medium"), font(24), font(56, "Bold"), font(30)


class Cam:
    """Close, fixed camera per episode, aimed between the robot base and the box."""
    def __init__(self, env):
        m = env.model
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        m.vis.quality.shadowsize = 4096
        self.env, self.r = env, mujoco.Renderer(m, height=VH, width=VW)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.62, 145, -20

    def aim(self):
        b = self.env.box_pos()
        self.cam.lookat[:] = [0.55 * b[0] + 0.03, 0.55 * b[1], 0.10]

    def shot(self):
        self.r.update_scene(self.env.data, self.cam)
        return self.r.render()


def dream_look(img):
    """Imagination: cool blue tint, slightly faded, soft vignette."""
    x = img.astype(np.float32) / 255
    gray = x.mean(2, keepdims=True)
    x = 0.55 * x + 0.45 * gray
    x = x * np.array([0.78, 0.9, 1.15]) + np.array([0.02, 0.05, 0.10])
    yy, xx = np.mgrid[0:x.shape[0], 0:x.shape[1]]
    r = np.hypot((xx - x.shape[1] / 2) / x.shape[1], (yy - x.shape[0] / 2) / x.shape[0])
    x *= (1 - 0.55 * np.clip(r - 0.25, 0, 1))[..., None]
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def set_dream(env, o):
    d, a = env.data, env.box_qpos_adr
    d.qpos[:6] = o[0:6]; d.qpos[a:a + 3] = o[15:18]; d.qpos[a + 3:a + 7] = quat_from_up(o[36:39])
    mujoco.mj_forward(env.model, d)


def compose(left, right, l_label, r_label, l_col, r_col, cap="", sub="", cap_alpha=1.0):
    im = Image.new("RGB", (W, H), BG)
    x0 = (W - 2 * VW - GAP) // 2
    for i, (v, lab, col) in enumerate([(left, l_label, l_col), (right, r_label, r_col)]):
        x = x0 + i * (VW + GAP)
        im.paste(Image.fromarray(v), (x, TOP))
        if lab:
            d = ImageDraw.Draw(im)
            tw = d.textlength(lab, font=F_LABEL)
            d.rounded_rectangle([x + 14, TOP + 14, x + 34 + tw, TOP + 50], 8, fill=(0, 0, 0))
            d.text((x + 24, TOP + 18), lab, font=F_LABEL, fill=col)
    if cap:
        c = tuple(int(BG[k] + cap_alpha * (v - BG[k])) for k, v in enumerate((245, 245, 245)))
        s = tuple(int(BG[k] + cap_alpha * (v - BG[k])) for k, v in enumerate((150, 156, 170)))
        d = ImageDraw.Draw(im)
        d.text((x0 + 4, TOP + VH + 22), cap, font=F_CAP, fill=c)
        if sub:
            d.text((x0 + 4, TOP + VH + 66), sub, font=F_SUB, fill=s)
    return np.asarray(im)


def card(title, body="", seconds=3.5):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    lines = body.split("\n") if body else []
    y = H // 2 - 40 - 22 * len(lines)
    d.text((110, y), title, font=F_TITLE, fill=(250, 250, 250))
    for i, line in enumerate(lines):
        d.text((112, y + 90 + 46 * i), line, font=F_BODY, fill=(160, 166, 180))
    return [np.asarray(im)] * int(seconds * FPS)


def policy(path):
    m = SAC.load(ROOT / path, device="cpu")
    return m, pickle.load(open(ROOT / path.replace(".zip", "_vecnorm.pkl"), "rb"))


def real_run(m, vn, seed):
    env = SO101PickEnv(); cam = Cam(env); o, _ = env.reset(seed=seed); cam.aim(); fr = [cam.shot()]
    for _ in range(150):
        o, _, _, _, info = env.step(m.predict(vn.normalize_obs(o), deterministic=True)[0])
        fr.append(cam.shot())
    return fr, info["is_success"]


def first_successes(m, vn, n=2):
    out = []
    for seed in range(10_000, 10_040):
        fr, ok = real_run(m, vn, seed)
        print(f"  seed {seed}: success {ok}", flush=True)
        if ok:
            out.append(fr)
        if len(out) == n:
            return out
    raise RuntimeError("not enough successes")


def dream_and_real(m, vn, wm, seed):
    real, dream = SO101PickEnv(), SO101PickEnv()
    cr, cd = Cam(real), Cam(dream)
    o, _ = real.reset(seed=seed); dream.reset(seed=seed); cr.aim(); cd.aim(); d = o.copy()
    L, R = [dream_look(cd.shot())], [cr.shot()]
    for _ in range(150):
        o, *_ = real.step(m.predict(vn.normalize_obs(o), deterministic=True)[0])
        a = m.predict(vn.normalize_obs(d), deterministic=True)[0]
        d = wm.predict(torch.tensor(d, dtype=torch.float32)[None], torch.tensor(a, dtype=torch.float32)[None]).mean(0)[0].numpy().astype(np.float64)
        set_dream(dream, d)
        L.append(dream_look(cd.shot())); R.append(cr.shot())
    return L, R, float(d[17] - 0.03), bool(pick_reward(d)[1])


def fade(a, b, n=10):
    return [((1 - t) * a.astype(np.float32) + t * b.astype(np.float32)).astype(np.uint8)
            for t in np.linspace(0, 1, n + 2)[1:-1]]


def section(L, R, l_lab, r_lab, l_col, r_col, cap, sub, hold=1.2):
    L, R = L + [L[-1]] * int(hold * FPS), R + [R[-1]] * int(hold * FPS)
    return [compose(l, r, l_lab, r_lab, l_col, r_col, cap, sub, min(1.0, i / 12)) for i, (l, r) in enumerate(zip(L, R))]


def join(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + fade(out[-1], p[0]) + p
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stills", action="store_true"); args = ap.parse_args()
    GOLD, BLUE, WHITE = (255, 205, 90), (140, 195, 255), (235, 235, 235)

    if args.stills:
        S = Path("/tmp/claude-1000/-home-rudra-naresh0201-projects-mcp/1e9e2591-437e-474f-82b8-7488c8829a84/scratchpad/frames")
        m, vn = policy("outputs/pickv2_warm.zip")
        a, _ = real_run(m, vn, 10_000)
        m5, vn5 = policy("outputs/dream_v5.zip"); _, _, wm = load("outputs/dream_v5.zip", "outputs/wm_v1.pt")
        L, R, lift, _ = dream_and_real(m5, vn5, wm, 10_000)
        for t in (0, 60, 110, 150):
            Image.fromarray(compose(L[t], R[t], "WORLD MODEL", "REALITY", BLUE, GOLD,
                                    "It learned to fool its own imagination.", "Left: what it imagines. Right: what actually happens.")).save(S / f"still_{t}.png")
        Image.fromarray(compose(a[0], a[120], "", "", WHITE, WHITE, "x", "y")).save(S / "still_sac.png")
        print("dream lift", lift); sys.exit()

    parts = [card("Can a robot learn by imagining?", "A simulated robot arm, a green box,\nand two ways to learn.", 4)]

    print("normal training")
    m, vn = policy("outputs/pickv2_warm.zip")
    a, b = first_successes(m, vn)
    parts.append(section(a, b, "", "", WHITE, WHITE, "Trained by trial and error, it picks up the box.",
                         "Two boxes it has never seen. In tests it succeeds every time."))

    parts.append(card("Practice is expensive.", "On a real robot every attempt takes time and wear.\n"
                      "So the robot learns how the world behaves,\nand practises in its imagination instead.", 5))

    print("the cheat")
    m, vn = policy("outputs/dream_v5.zip"); _, _, wm = load("outputs/dream_v5.zip", "outputs/wm_v1.pt")
    for seed in range(10_000, 10_040):
        L, R, lift, g = dream_and_real(m, vn, wm, seed)
        print(f"  seed {seed}: imagined lift {100 * lift:.1f} cm", flush=True)
        if lift > 0.10:
            break
    parts.append(section(L, R, "WORLD MODEL", "REALITY", BLUE, GOLD, "It learned to fool its own imagination.",
                         "Left: it imagines grabbing the box and lifting it. Right: it never touches it.", 1.8))

    parts.append(card("So we kept it honest.", "A little real experience to correct the imagination,\n"
                      "caution where the imagination is unsure,\nand one rule: you can't grip a box from 6 cm away.", 5))

    print("after the fix")
    m, vn = policy("outputs/ckpt/dream_v7_s1_300k.zip")
    a, b = first_successes(m, vn)
    parts.append(section(a, b, "", "", WHITE, WHITE, "Trained mostly in imagination, it now picks up the box.",
                         "6 in 10 boxes at this point. The normally trained robot gets 10 in 10."))

    parts.append(card("Not solved yet.", "It works in 2 of 3 training runs, and it sometimes forgets what it learned.\n"
                      "Next: make it reliable, then learn from a camera.", 5))

    V = join(parts)
    out = ROOT / "outputs/robot_story.mp4"
    iio.imwrite(out, np.stack(V), fps=FPS, codec="libx264", macro_block_size=8, pixelformat="yuv420p", quality=8)
    print("wrote", out, f"({len(V) / FPS:.0f} s)")
