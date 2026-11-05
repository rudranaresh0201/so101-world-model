"""Tests YOUR envs/so101_reach.py, one TODO at a time.
    python scripts/check_env.py
Fix the first FAIL, re-run, repeat. When everything passes, you're ready to train.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except NotImplementedError as e:
        print(f"  TODO  {name}: {e}")
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
    except Exception:
        print(f"  FAIL  {name}: crashed\n" + "".join("        " + line for line in traceback.format_exc().splitlines(True)[-4:]))
    return False


from envs.so101_reach import BOX_X_RANGE, BOX_Y_RANGE, SO101ReachEnv  # noqa: E402

env = SO101ReachEnv()
state = {}


def t_reset():
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict), "reset must return (obs, info_dict)"
    state["obs"] = obs


def t_obs():
    obs = state["obs"]
    assert obs.shape == (27,), f"observation should have 27 numbers, got shape {obs.shape}"
    assert np.all(np.isfinite(obs)), "observation contains NaN/inf"
    claw, box, diff = obs[12:15], obs[15:18], obs[18:21]
    assert np.allclose(claw, env.claw_pos()), "obs[12:15] should be the claw position"
    assert np.allclose(box, env.box_pos()), "obs[15:18] should be the box position"
    assert np.allclose(diff, env.target_pos() - claw), "obs[18:21] should be target - claw (not claw - target!)"


def t_random_box():
    spots = []
    for s in range(20):
        env.reset(seed=s)
        x, y, _ = env.box_pos()
        assert BOX_X_RANGE[0] <= x <= BOX_X_RANGE[1] and BOX_Y_RANGE[0] <= y <= BOX_Y_RANGE[1], \
            f"box at x={x:.3f}, y={y:.3f} is outside the allowed range"
        spots.append((round(x, 4), round(y, 4)))
    assert len(set(spots)) > 15, "box lands in the same spot every reset -- not random?"
    env.reset(seed=7)
    a = env.box_pos()
    env.reset(seed=7)
    assert np.allclose(a, env.box_pos()), "same seed gave different boxes -- use self.np_random, not np.random"
    assert np.allclose(env.box_start, env.box_pos()), "self.box_start should equal the box position after reset"
    assert env.steps == 0, "step counter should be 0 after reset"


def t_delta_action():
    env.reset(seed=1)
    before = env.data.ctrl.copy()
    env.step(np.ones(6, dtype=np.float32))
    change = env.data.ctrl - before
    assert np.allclose(change, 0.05), f"a full +1 action should raise every target by 0.05 rad, got {change.round(3)}"
    for _ in range(200):  # push far past the joint limits
        env.step(np.ones(6, dtype=np.float32))
        if env.steps >= 99:
            break
    assert np.all(env.data.ctrl <= env.ctrl_high + 1e-9), "targets went past the joint limits -- clip them"


def t_physics_moves():
    env.reset(seed=2)
    q0 = env.data.qpos[:6].copy()
    env.step(np.ones(6, dtype=np.float32))
    assert not np.allclose(q0, env.data.qpos[:6]), "joints didn't move after a step -- did you run mj_step?"
    assert abs(env.data.time - 0.05) < 1e-6, f"one decision should simulate 0.05 s, simulated {env.data.time:.3f} s"


def t_reward_and_endings():
    env.reset(seed=3)
    rewards, ended_by = [], None
    for _ in range(150):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert 0.0 <= r <= 1.0, f"reward {r} outside [0, 1]"
        expected = 1 - np.tanh(5 * info["distance"])
        assert abs(r - expected) < 1e-6, f"reward {r:.4f} != 1 - tanh(5*distance) = {expected:.4f}"
        rewards.append(r)
        if term or trunc:
            ended_by = "terminated" if term else "truncated"
            break
    assert ended_by is not None, "episode never ended in 150 steps -- time limit should stop it at 100"
    if ended_by == "truncated":
        assert env.steps == 100, f"truncated at step {env.steps}, expected 100"
    print(f"        (random episode: {len(rewards)} steps, mean reward {np.mean(rewards):.3f}, ended: {ended_by})")


def t_knock():
    env.reset(seed=4)
    adr = env.box_qpos_adr
    env.data.qpos[adr] += 0.08  # shove the box 8 cm sideways
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    _, _, term, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["knocked"] and term, "box moved 8 cm sideways but the episode didn't terminate as knocked"


def t_gym_checker():
    from gymnasium.utils.env_checker import check_env
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(SO101ReachEnv(), skip_render_check=True)


print("Checking envs/so101_reach.py\n")
results = [
    check("TODO 2  reset() runs", t_reset),
    check("TODO 1  observation layout", t_obs) if "obs" in state else print("  SKIP  observation (reset failed)"),
    check("TODO 2  box randomised, seeded, start remembered", t_random_box),
    check("TODO 3a delta action + joint-limit clipping", t_delta_action),
    check("TODO 3b physics actually runs (10 steps = 0.05 s)", t_physics_moves),
    check("TODO 3c/d reward formula + time limit", t_reward_and_endings),
    check("TODO 3d knocked box terminates", t_knock),
    check("Gymnasium's own API checker", t_gym_checker),
]
ok = all(r is True for r in results)
print("\n" + ("ALL PASS -- ready to train: python train/train_reach.py" if ok else "Not yet. Fix the first TODO/FAIL above and re-run."))
