"""Lesson: what is inside a MuJoCo model?  python scripts/inspect_model.py

A MuJoCo simulation has two objects:
  model (mjModel) -- the FIXED description: bodies, joints, actuators, masses. Loaded from XML.
  data  (mjData)  -- the CHANGING state: positions (qpos), velocities (qvel), motor commands (ctrl).
One physics step:  mujoco.mj_step(model, data)  advances data by model.opt.timestep seconds.
"""
from pathlib import Path

import mujoco

SCENE = Path(__file__).resolve().parents[1] / "third_party/mujoco_menagerie/robotstudio_so101/scene_box.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

print(f"timestep: {model.opt.timestep}s  ->  {1 / model.opt.timestep:.0f} physics steps per simulated second")
print(f"nq (size of qpos) = {model.nq}   nv (size of qvel) = {model.nv}   nu (number of actuators) = {model.nu}\n")

print("JOINTS (each owns a slice of qpos):")
for j in range(model.njnt):
    jtype = ["free", "ball", "slide", "hinge"][model.jnt_type[j]]
    adr = model.jnt_qposadr[j]
    lo, hi = model.jnt_range[j]
    rng = f"range [{lo:+.2f}, {hi:+.2f}] rad" if model.jnt_limited[j] else "unlimited"
    print(f"  {model.joint(j).name or '(box freejoint)':16} {jtype:6} qpos[{adr}]  {rng}")

print("\nACTUATORS (what a policy controls -- data.ctrl):")
for a in range(model.nu):
    lo, hi = model.actuator_ctrlrange[a]
    target = model.joint(model.actuator_trnid[a, 0]).name
    kp = model.actuator_gainprm[a, 0]
    print(f"  ctrl[{a}] {model.actuator(a).name:14} drives joint {target:14} target range [{lo:+.2f}, {hi:+.2f}]  kp={kp:g}")

print("\nThese are POSITION actuators: ctrl is a target joint angle, and a built-in PD controller")
print("pushes the joint toward it. So a policy's action = 6 target angles, not raw motor torques.")

mujoco.mj_forward(model, data)
print("\nBox position (xyz, metres):", data.body("box").xpos.round(3))
print("Gripper site / end-effector candidates:", [model.site(i).name for i in range(model.nsite)])
