# SO-101 × World Models: the plan

**Goal:** teach a simulated SO-101 arm to pick up a box with RL in MuJoCo. Then train it inside a
learned world model (the "dream"), show on video where the dream lies and the policy exploits it,
and fix that. Optionally, finish on a real arm.

**Deliverables:** a split-screen demo video (dream vs. reality), a blog post, and possibly a paper.

**How we work:** you write the code and I explain, review and debug. Nothing is "done" until you
can explain it without notes.

---

## 0. Setup (done)

```bash
cd ~/projects/so101 && source .venv/bin/activate
export MUJOCO_GL=egl                       # offscreen rendering (no GPU needed)
python scripts/inspect_model.py            # what's inside the robot model
python scripts/render.py --key pickup      # writes outputs/pickup.png
python scripts/view.py                     # interactive viewer: drag the Control sliders
```

| Path | What |
|---|---|
| `third_party/mujoco_menagerie/robotstudio_so101/` | official SO-101 model (`so101.xml`, `scene_box.xml`) |
| `scripts/` | tools: inspect, render, view |
| `exercises/` | practice tasks with TODOs **for you** |
| `envs/` | **empty**: our SO-101 Gymnasium env goes here (you write it, Stage 4) |
| `notes/` | your notes, one file per stage |

Facts already learned from `inspect_model.py`:
- Physics runs at 200 steps/s (timestep 0.005 s). `qpos` has 13 numbers: 6 arm joints plus 7 for the
  box's free joint (xyz position + quaternion).
- 6 **position actuators**: `ctrl` is a *target angle* and a built-in PD controller (kp≈998) drives the
  joint to it, so a policy outputs 6 target angles, not torques.
- In the default scene the box is at x=0.5 m but the fully stretched gripper only reaches about 0.39 m,
  so **our env must spawn the box within reach**.

---

## 1. Revise what you already know (day 1, ~2 h)

You've done basic RL and world models, so refresh rather than relearn. For each item, write 3–5
lines in `notes/00-revision.md`.
- **MDP:** state, action, reward, transition, discount γ. What's a *return*?
- **Policy π(a|s), value V(s), Q(s,a), advantage A = Q − V.** Why does the advantage reduce variance?
- **On-policy vs. off-policy.** Why can SAC reuse old data but PPO can't?
- **Exploration vs. exploitation**, and what entropy has to do with it.
- **World model:** encoder → latent state → dynamics → decoder/reward head. What does "imagination" mean?

Reference: [Spinning Up, "Key Concepts in RL"](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).

---

## 2. Stages: learn → build → check

### Stage 1: RL fundamentals, hands-on
- **Read:** [HF Deep RL Course](https://huggingface.co/learn/deep-rl-course), units 1–4 and 8 (PPO).
  [Spinning Up](https://spinningup.openai.com), Intro parts 1–3.
- **Optional depth:** Sutton & Barto, *RL: An Introduction*, chapters 1, 3 and 6 (free PDF).
- **Do:** `exercises/ex1_cartpole_ppo.py`, all three TODOs. (Verified here: about 1.5 min, reaches 500/500.)
- **Check:** explain PPO's clipped objective and *why* clipping stabilises training.

### Stage 2: continuous control (the RL a robot actually uses)
- **Read:** Spinning Up's pages on [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
  and [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
- **Read the code:** [CleanRL](https://github.com/vwxyzjn/cleanrl) `cleanrl/sac_continuous_action.py`,
  the whole algorithm in one file. Annotate every block in `notes/02-sac.md`.
- **Do:** `exercises/ex2_pendulum_sac.py`, then with `Reacher-v5` (a 2-joint arm, a mini version of
  our task). SAC is slow on this CPU: about 10+ min.
- **Check:** what are SAC's three networks, why are there two Q-networks, and what does α do?

### Stage 3: MuJoCo itself
- **Read:** [MuJoCo docs](https://mujoco.readthedocs.io): "Overview" (all of it), then "Modeling"
  (bodies, joints, geoms, actuators). Skim "Computation".
- **Do:** the official tutorial notebook, `python/tutorial.ipynb` in
  [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco).
- **Do:** read `so101.xml` top to bottom. Open `scripts/view.py` and move each joint with the sliders.
  Find the `gripperframe` site.
- **Do (you write it):** `exercises/ex3_scripted_pick.py`, a *hand-coded* (no RL) script that moves
  the arm to the box, closes the gripper and lifts. Hint: start from the `pickup` keyframe's `ctrl`
  values. This teaches you control and contacts before RL hides them.
- **Check:** what happens inside one `mj_step`? What's the difference between `qpos`, `ctrl` and
  `xpos`? Why does the gripper slip if friction is too low?

### Stage 4: robot RL in practice (Milestone 1, our first real build)
- **Read:** Lilian Weng, ["Domain Randomization for Sim2Real Transfer"](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/).
  Tobin et al. 2017, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World".
- **Skim:** [LeRobot docs](https://huggingface.co/docs/lerobot), the SO-101 pages.
- **Build, in `envs/so101_reach.py` (you write it, I review):**
  1. A Gymnasium `Env`: `reset()` randomises the box position *within reach*; `step(action)` maps
     6 numbers in [-1, 1] to joint targets and runs N physics steps.
  2. Observation: joint angles, joint velocities, gripper position, box position.
  3. **Reach** task first: reward = −distance(gripper, box). Train SAC. Watch the video.
  4. Then **lift**: shaped reward (reach + grasp + height). Expect it to fail at first. Debugging the
     reward is the lesson.
- **Check:** why did your first reward fail, and what did you change? What's *action repeat*, and
  why does it help?

### Stage 5: world models
- **Read:**
  1. Ha & Schmidhuber, ["World Models"](https://worldmodels.github.io): intuition.
  2. [DreamerV3](https://danijar.com/project/dreamerv3/) project page + paper §1–3. The RSSM and
     "learning behaviours in imagination".
  3. **DayDreamer** (Wu et al., 2022): Dreamer on *real* robots, learning in hours.
  4. **MBPO**, "When to Trust Your Model" (Janner et al., 2019): the classic on model exploitation.
     Short rollouts, ensembles.
  5. **TD-MPC2** (Hansen et al.): the other school, planning in latent space.
- **Build (Milestone 2):** a small world model on our SO-101 env (an MLP ensemble predicting the next
  state and reward, MBPO-style) before anything image-based. Train the policy partly or fully on
  imagined rollouts. Compute moves to a cloud GPU here.
- **Check:** how does Dreamer train an actor without touching the environment, and exactly where
  can that go wrong?

### Stage 6: the research part (Milestones 3–4)
- **Read the 2026 frontier:**
  - [Imperfect World Models are Exploitable](https://www.alphaxiv.org/abs/2605.15960): a formal
    definition of exploitation.
  - [World Models Need Not Be Right to Rank Policies Right](https://arxiv.org/html/2611.01139):
    ranking vs. exploitation, audits.
  - [RENEW](https://arxiv.org/html/2607.14180v1): repairing exploitation with *human* preferences.
  - [Survey: World Models for Robot Learning](https://arxiv.org/html/2605.00080v1): the map.
  - [Dreamer "delusion gap" tutorial](https://zero2robot.com/ch3.2-dreamer/): the exact gap, hands-on.
- **Milestone 3, the gap:** replay the policy's imagined rollouts in real MuJoCo and record dream
  return vs. real return. Make the **split-screen video**. Find *where* they diverge (contacts? grasp?
  out-of-distribution states?).
- **Milestone 4, the fix:** detect exploitation (ensemble disagreement, dream return rising while
  real return stays flat) and spend a small real-interaction budget exactly there. The headline
  question: *how much real data stops a policy from cheating in its dreams?*

### Stage 7 (optional): the real arm
- Domain randomisation plus a little real data, deployed on a real SO-101. A real-robot clip is the
  strongest possible demo.

---

## 3. Compute
- Stages 1–4 run on this CPU (no GPU, 6 GB RAM). Keep runs small.
- Stage 5+ needs a cloud GPU (Colab Pro / RunPod / Lambda), costing a few dollars per run.

## 4. Order of work (tonight onward)
1. **Revision:** fill in `notes/00-revision.md` (section 1).
2. **Stage 1:** run `ex1`, do its TODOs, explain PPO clipping to Claude.
3. **Stage 3, first half:** `inspect_model.py` + `view.py`; read `so101.xml` and move the joints by hand.
4. **Stage 2:** SAC on Pendulum → Reacher, reading CleanRL's SAC alongside.
5. Write `ex3_scripted_pick.py` (hand-coded pick), then `envs/so101_reach.py`. **Milestone 1.**
