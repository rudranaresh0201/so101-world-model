"""Text-free version of the story video: only the simulation, no titles, captions or labels.
The explanation lives in the tweet/post text instead.

    MUJOCO_GL=egl python wm/clean_video.py   -> outputs/robot_story_clean.mp4

1. normal training: two unseen boxes, both picked
2. the cheat: world model (blue, faded "ghost" world) left, reality right
3. after the fixes: the dream-trained policy picks two boxes for real
Same clip selection as wm/story_video.py (first test boxes showing the behaviour).
"""
import sys
from pathlib import Path
import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wm.story_video as sv  # noqa: E402

W, H, GAP = 1280, 720, 8
sv.VW, sv.VH = (W - GAP) // 2, H          # full-height views, no caption area
BG = np.array(sv.BG, np.uint8)


def pair(L, R, hold=1.0):
    L, R = L + [L[-1]] * int(hold * sv.FPS), R + [R[-1]] * int(hold * sv.FPS)
    gap = np.broadcast_to(BG, (H, GAP, 3))
    return [np.concatenate([l, gap, r], axis=1) for l, r in zip(L, R)]


def black_fade(frames, n=12):
    black = np.zeros_like(frames[0])
    return sv.fade(black, frames[0], n) + frames + sv.fade(frames[-1], black, n)


if __name__ == "__main__":
    print("normal training")
    m, vn = sv.policy("outputs/pickv2_warm.zip")
    a, b = sv.first_successes(m, vn)
    part1 = pair(a, b)

    print("the cheat")
    m, vn = sv.policy("outputs/dream_v5.zip"); _, _, wm = sv.load("outputs/dream_v5.zip", "outputs/wm_v1.pt")
    for seed in range(10_000, 10_040):
        L, R, lift, _ = sv.dream_and_real(m, vn, wm, seed)
        print(f"  seed {seed}: imagined lift {100 * lift:.1f} cm", flush=True)
        if lift > 0.10:
            break
    part2 = pair(L, R, 1.5)

    print("after the fix")
    m, vn = sv.policy("outputs/ckpt/dream_v7_s1_300k.zip")
    a, b = sv.first_successes(m, vn)
    part3 = pair(a, b)

    V = black_fade(sv.join([part1, part2, part3]))
    out = ROOT / "outputs/robot_story_clean.mp4"
    iio.imwrite(out, np.stack(V), fps=sv.FPS, codec="libx264", macro_block_size=8, pixelformat="yuv420p", quality=8)
    print("wrote", out, f"({len(V) / sv.FPS:.0f} s)")
