"""Feasibility test for PICK: scripted 6-D IK (position + 'fingers point down' orientation).
Open -> hover above box -> descend so the jaws straddle it -> close -> lift.  No learning."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import mujoco, numpy as np
from envs.so101_reach import SO101ReachEnv, FRAME_SKIP

e = SO101ReachEnv(); m, d = e.model, e.data
sid = m.site("gripperframe").id
OPEN, CLOSED = 1.2, -0.17
# site frame: fingers point along local +x, jaws close along local z (fixed jaw on the -z side)
H = np.array([1.0, 0, 0])                               # jaw-closing direction in the world
R_DES = np.column_stack([[0, 0, -1], np.cross(H, [0, 0, -1]), H])   # columns = desired local x, y, z

def go(goal, grip, steps, w_rot=0.3):
    for _ in range(steps):
        R = d.site_xmat[sid].reshape(3, 3)
        e_pos = goal - d.site_xpos[sid]
        e_rot = 0.5 * sum(np.cross(R[:, i], R_DES[:, i]) for i in range(3))
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp[:, :5], w_rot * jacr[:, :5]]); err = np.concatenate([e_pos, w_rot * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        d.ctrl[:5] = np.clip(d.qpos[:5] + np.clip(dq, -0.1, 0.1), e.ctrl_low[:5], e.ctrl_high[:5])
        d.ctrl[5] = grip
        for _ in range(FRAME_SKIP): mujoco.mj_step(m, d)

def trial(seed, z_grasp, frames=None):
    e.reset(seed=seed)
    box0 = e.box_pos(); grasp = box0 - 0.01 * H           # site sits ~on the fixed jaw: shift so jaws straddle
    go(grasp + [0, 0, 0.10], OPEN, 40)
    go(grasp + [0, 0, z_grasp], OPEN, 40)
    pre = e.box_pos()
    go(grasp + [0, 0, z_grasp], CLOSED, 20)
    go(grasp + [0, 0, 0.15], CLOSED, 40)
    tilt = np.degrees(np.arccos(np.clip(-d.site_xmat[sid].reshape(3, 3)[2, 0], -1, 1)))
    return e.box_pos()[2] - 0.03, np.linalg.norm(pre[:2] - box0[:2]), tilt

if __name__ == "__main__":
    for z in [-0.01, 0.0, 0.01, 0.02]:
        r = np.array([trial(10_000 + i, z) for i in range(20)])
        print(f"grasp height {100*z:+.0f} cm: lifted>5cm {np.mean(r[:,0] > 0.05):4.0%}   median lift {100*np.median(r[:,0]):5.1f} cm"
              f"   push before close {100*np.median(r[:,1]):4.1f} cm   finger tilt from vertical {np.median(r[:,2]):4.0f} deg")
