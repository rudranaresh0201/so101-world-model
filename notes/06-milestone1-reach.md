# Milestone 1: reach is solved

Config D: observation normalisation, and a knocked box does not end training episodes. 200k SAC steps, 2 seeds.
The evaluation uses 30 unseen boxes and stops the episode when the box gets knocked.

| seed | success @100k | success @200k | knocked @200k | median final distance |
|------|---------------|---------------|---------------|-----------------------|
| 1    | 63%           | 93%           | 10%           | 1.2 cm                |
| 2    | 57%           | 90%           | 7%            | 1.0 cm                |

Both seeds land at the same place, so the result is not a lucky seed. The curve is noisy (drops at 110k and 190k).
That is normal for off-policy RL with a 30-episode evaluation.
Video: `python train/watch.py --name D_long_s1` (6/6 on fresh boxes, 0.3–0.8 cm).

## The whole debugging chain (what each failure taught)
1. Target was the box centre, which the claw can never reach. **Check feasibility first.**
2. Hidden wound-up motor targets made the state non-Markov. **The obs must contain what drives the dynamics.**
3. Success ended the episode while every step paid positive reward. **Termination is part of the reward.**
4. Knock termination plus raw obs scales: the agent learned to stay away. **Ablate one thing at a time.**
