# SO-101 world model

SAC teaches a simulated SO-101 arm to pick up a box in MuJoCo. A second SAC policy then trains inside a
learned world model instead of the simulator, and is always tested back in real MuJoCo. The question is
whether training in the model transfers, and how the policy exploits it when it doesn't.

## Results

| Setup | Real success |
|---|---|
| SAC in MuJoCo (baseline) | 100%, 2 seeds |
| SAC in the world model, 5-step rollouts | 0%. The same recipe in real MuJoCo also fails, so the short horizon is the cause, not the model |
| SAC in the world model, 50-step rollouts | 0%. The model says the box was grasped 100% of the time: the policy exploits it |
| 50-step rollouts + real-data loop + disagreement penalty + 6 cm grasp check | picks on 2 of 3 seeds, best 60%, unstable |

Rollout horizon, in real MuJoCo: 5 steps fails, 20 reaches 85%, 50 reaches 95%.

**The exploit.** The policy hovers about 16 cm from the box while the model's predicted contact signals drift
upward. The reward reads those signals, counts a grasp, and the imagined box rises up to 50 cm. In reality
the box never moves.

**The fixes.**
- Every 5k dream steps, 4 real episodes are added to the model's data and to SAC's replay buffer.
- The reward is penalised by ensemble disagreement.
- A grasp needs the claw within 6 cm of the box. The largest distance in 91k real grasps was 5.1 cm, so this
  changes no real reward.

**Not shown yet.** That the world model saves real steps. Its data came from already-trained policies; a fair
test needs a model built from random play only, with every real step counted.

## Layout

| Path | What |
|---|---|
| `envs/` | Gymnasium envs: `so101_reach.py`, `so101_pick.py` (41-number state, 6 joint-delta actions, staged reward) |
| `train/` | SAC training (`train_pick.py`), evaluation, video rendering |
| `wm/collect.py` | records transitions from a mix of random and trained policies |
| `wm/model.py`, `wm/train_wm.py` | 5-member MLP ensemble predicting the state change; training and open-loop evaluation |
| `wm/reward.py` | the pick reward computed exactly from a predicted state, so there is no reward head |
| `wm/dream_sac.py` | SAC inside the model (MBPO-style rollouts from real start states) |
| `wm/dream_v3.py` | the online loop with real data, penalty and grasp check |
| `wm/short_real.py` | the same rollout recipe in real MuJoCo, for the horizon ablations |
| `wm/clean_video.py` | the demo video |
| `notes/` | post-mortems from the reach and pick stages |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie checkout 98a70c5
export MUJOCO_GL=egl   # offscreen rendering

python train/train_pick.py --name pickv2                 # baseline SAC
python wm/collect.py && python wm/train_wm.py            # dataset, world model
python wm/dream_v3.py --name dream_v7 --horizon 50 --penalty 1.0 --consistent --real_mix
```

Everything runs on a laptop CPU. The state-based world model trains in about 15 minutes.
