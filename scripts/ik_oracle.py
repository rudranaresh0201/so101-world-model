"""Feasibility test: a scripted Jacobian controller using the SAME action interface as the policy."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path.home() / "projects/so101"))
import mujoco, numpy as np
from envs.so101_reach import SO101ReachEnv, MAX_DELTA
e = SO101ReachEnv(); m, d = e.model, e.data
sid = m.site("gripperframe").id
res = []
for ep in range(30):
    e.reset(seed=10_000 + ep)
    best = 9; knocked = False
    for t in range(100):
        err = e.target_pos() - e.claw_pos()
        jacp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:, :5]                                   # arm joints (no gripper)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)   # damped least squares
        dq_des = np.zeros(6); dq_des[:5] = dq
        # desired NEW target = current angle + dq  -> express as a delta on ctrl
        want = d.qpos[:6] + dq_des
        a = np.clip((want - d.ctrl) / MAX_DELTA, -1, 1).astype(np.float32)
        _, _, term, trunc, info = e.step(a)
        best = min(best, info["distance"])
        if term or trunc:
            knocked = info["knocked"]; break
    res.append((info["is_success"], knocked, info["distance"], best))
res = np.array(res, dtype=float)
print(f"scripted IK oracle: success {res[:,0].mean():.0%}  knocked {res[:,1].mean():.0%}  "
      f"median final {100*np.median(res[:,2]):.1f} cm  median closest {100*np.median(res[:,3]):.1f} cm")
fails = [(i, round(100*r[2],1)) for i, r in enumerate(res) if not r[0]]
print("failures (episode, final cm):", fails[:12])
