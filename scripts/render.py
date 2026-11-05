"""Render the SO-101 scene to a PNG without opening a window.
    python scripts/render.py                -> home pose
    python scripts/render.py --key pickup   -> the scene's saved 'pickup' keyframe
Writes outputs/<name>.png
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "third_party/mujoco_menagerie/robotstudio_so101/scene_box.xml"

ap = argparse.ArgumentParser()
ap.add_argument("--key", default=None)
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
if args.key:
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.key).id)
mujoco.mj_forward(model, data)

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.25, 0.0, 0.1]
cam.distance, cam.azimuth, cam.elevation = 0.9, 140, -25
with mujoco.Renderer(model, height=480, width=640) as r:
    r.update_scene(data, camera=cam)
    img = r.render()
out = ROOT / "outputs" / f"{args.key or 'home'}.png"
iio.imwrite(out, img)
print("gripper at", data.site("gripperframe").xpos.round(3), "| box at", data.body("box").xpos.round(3))
print("wrote", out)
