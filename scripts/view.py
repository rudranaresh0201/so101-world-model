"""Open the interactive MuJoCo viewer on the SO-101 scene.
    python scripts/view.py
Double-click a body to select it, Ctrl+right-drag to push it. The right panel's
'Control' sliders set data.ctrl -- drag them to move each joint by hand.
"""
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parents[1] / "third_party/mujoco_menagerie/robotstudio_so101/scene_box.xml"
model = mujoco.MjModel.from_xml_path(str(SCENE))
mujoco.viewer.launch(model)
