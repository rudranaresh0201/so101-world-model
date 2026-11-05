"""Render one picture per joint: each panel moves ONE joint and leaves the rest at 0.
    python scripts/joint_tour.py  -> outputs/joint_tour.png
"""
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
model = mujoco.MjModel.from_xml_path(str(ROOT / "third_party/mujoco_menagerie/robotstudio_so101/scene_box.xml"))
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.15, 0.0, 0.15]
cam.distance, cam.azimuth, cam.elevation = 0.85, 120, -20

moves = [("all zero (home)", None, 0), ("shoulder_pan = +0.8", 0, 0.8), ("shoulder_lift = +0.8", 1, 0.8),
         ("elbow_flex = -0.8", 2, -0.8), ("wrist_flex = +1.0", 3, 1.0), ("gripper = 1.5 (open)", 5, 1.5)]
panels = []
with mujoco.Renderer(model, height=300, width=400) as r:
    for label, j, val in moves:
        mujoco.mj_resetData(model, data)
        if j is not None:
            data.qpos[j] = val
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        img = r.render().copy()
        panels.append(img)
        print(f"{label:24} -> gripper tip at {data.site('gripperframe').xpos.round(2)}")
grid = np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0)
iio.imwrite(ROOT / "outputs/joint_tour.png", grid)
